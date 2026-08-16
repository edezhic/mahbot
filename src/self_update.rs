//! Self-update logic — single-instance guarding, build, binary swap, and restart.
//!
//! Two update modes are supported, selected at runtime by [`update_mode`]:
//!
//! * **Local checkout** (the historical mode): the binary was built from a
//!   local source checkout (`.git` present at `CARGO_MANIFEST_DIR`, or a plain
//!   source tree with `Cargo.toml`). Update = `cargo build --release` →
//!   [`self_replace`](https://docs.rs/self-replace) → copy to cargo install
//!   bin → shutdown → checkpoint → restart.
//! * **Registry** (crates.io-installed binaries): `CARGO_MANIFEST_DIR` points
//!   into the cargo registry/git source cache (e.g. `~/.cargo/registry/src/`),
//!   so a local rebuild would be a same-version no-op. Update = periodic
//!   crates.io sparse-index check for a strictly newer stable version →
//!   `cargo install <crate> --force` → shutdown → checkpoint → restart.
//!
//! Uses `flock()` for single-instance enforcement (kernel guarantees lock release on
//! process death). The update flow: build/install → swap → copy to cargo install bin
//! → shutdown agents → checkpoint all Turso databases
//! (the last single-writer checkpoint, while the instance lock is still held) →
//! release lock → spawn new instance from cargo bin path (or `current_exe()`
//! fallback) → remove build artifact (guarded against deleting the spawn
//! target, `current_exe()`, or the cargo bin path) → `exit(0)`.
//!
//! The cargo install path resolution uses `$CARGO_HOME` if set, else
//! `~/.cargo/bin` via `directories::UserDirs` — this ensures the
//! self-updated binary is visible to the shell tool and the user's PATH
//! (the shell tool's `extra_shell_path_prefixes` includes both paths when
//! they differ, so the single resolved path is always covered).
//!
//! The WAL checkpoint before `exit(0)` is a clean store handoff:
//! `std::process::exit(0)` bypasses all Rust destructors, so Turso connections
//! are never properly closed. The TRUNCATE leaves a header-only WAL with the
//! shared frame index reset; committed data is already fsync-durable at COMMIT.
//!
//! ## macOS Gatekeeper safety
//!
//! `posix_spawn` triggers async Gatekeeper code-signing validation; deleting the
//! spawn target during validation produces empty stderr (SIGKILL by `syspolicyd`).
//! See `should_delete_build_artifact()` and `execute_update()` steps 13–14.

use crate::util::is_executable;
use anyhow::{Context, Result, anyhow};
#[cfg(test)]
use directories::UserDirs;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::OnceLock;
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
    let lock_path = crate::lock_utils::lock_file_path(storage_root);

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

    if crate::lock_utils::try_flock(&file)
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
pub enum UpdateMode {
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
pub fn update_mode() -> UpdateMode {
    classify_update_mode(Path::new(env!("CARGO_MANIFEST_DIR")))
}

/// Whether a self-update is available on this installation, independent of
/// whether a newer version exists.
///
/// - Local checkout: the build checkout (with `Cargo.toml`) is reachable.
/// - Registry: non-Windows (the platform can run the update). Note this does
///   NOT verify that `cargo` is on PATH — that prerequisite is checked
///   inside [`execute_registry_update`], which reports a proper error if it
///   is missing. Registry self-update is not offered on Windows (a running
///   `.exe` cannot be replaced), matching the local-checkout behavior of
///   hiding the button.
#[must_use]
pub fn is_update_available() -> bool {
    match update_mode() {
        UpdateMode::LocalCheckout => Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("Cargo.toml")
            .is_file(),
        UpdateMode::Registry => !cfg!(windows),
    }
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
/// Windows: always `Ok(None)` — the running `.exe` cannot be replaced, so the
/// registry update is not offered there. The GUI never calls this on Windows
/// (the subscription is wired only for non-Windows), but the guard is kept as
/// public-API safety so a stray caller cannot discover an update that cannot
/// be installed.
pub async fn check_registry_update() -> Result<Option<semver::Version>> {
    if cfg!(windows) {
        return Ok(None);
    }
    let Some(latest) = fetch_latest_stable_version().await? else {
        return Ok(None);
    };
    let current = semver::Version::parse(VERSION)
        .with_context(|| format!("embedded version {VERSION} is not valid semver"))?;
    Ok((latest > current).then_some(latest))
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
static UPDATE_FINALIZING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// True while [`execute_update`] is in its finalizing window (daemon shut
/// down; checkpoint, lock release, spawn, and `exit(0)` pending).
#[must_use]
pub fn update_is_finalizing() -> bool {
    UPDATE_FINALIZING.load(std::sync::atomic::Ordering::SeqCst)
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
/// - [`UpdateMode::LocalCheckout`]: build from source, swap binary, install to
///   cargo bin, notify admin, restart.
/// - [`UpdateMode::Registry`]: `cargo install <crate> --force` from crates.io,
///   notify admin, restart.
///
/// Called from the GUI update button.
/// Only one update runs at a time — concurrent calls return an error immediately.
///
/// On success, this function never returns (`std::process::exit(0)`).
/// On failure, returns an error.
pub(crate) async fn execute_update() -> Result<()> {
    match update_mode() {
        UpdateMode::LocalCheckout => execute_local_update().await,
        UpdateMode::Registry => execute_registry_update().await,
    }
}

/// Local-checkout self-update: rebuild from the source checkout, swap the
/// running binary, install to the cargo bin path, restart.
///
/// See [`execute_update`] for the concurrent-guard and exit contracts.
async fn execute_local_update() -> Result<()> {
    // Concurrent guard — only one update at a time.
    let Some(_guard) = UPDATE_MUTEX.try_lock().ok() else {
        anyhow::bail!("An update is already in progress. Please wait for it to complete.");
    };

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

    // 4. Compute paths early (needed by copy, spawn, and cleanup below).
    let binary_path = manifest_dir.join("target").join("release").join({
        let exe_suffix = std::env::consts::EXE_SUFFIX;
        if exe_suffix.is_empty() {
            "mahbot".to_string()
        } else {
            format!("mahbot{exe_suffix}")
        }
    });

    // Resolve the cargo install bin path. Checks `$CARGO_HOME` first, then
    // falls back to `~/.cargo/bin` via `directories::UserDirs`. Unlike the
    // shell tool's `extra_shell_path_prefixes` (which adds both paths as a
    // belt-and-suspenders measure), this function returns a single path —
    // whichever is selected is guaranteed to be present in the shell PATH
    // since `extra_shell_path_prefixes` includes both.
    let cargo_bin_path = resolve_cargo_bin_path();

    // 5. Run cargo build.
    run_cargo_with_timeout(
        &["build", "--release", "--locked"],
        manifest_dir,
        std::time::Duration::from_mins(30),
        "cargo build --release",
        "Build",
        admin_target.as_deref(),
    )
    .await?;

    // 6. self_replace — swap the running binary with the newly built one.
    self_replace::self_replace(&binary_path)
        .with_context(|| format!("Failed to swap binary at {}", binary_path.display()))?;

    // 7. Resolve the spawn target: copy the new binary to the cargo install path
    //    (so the shell tool's PATH resolution finds it), or use current_exe() as
    //    fallback. The copy is skipped entirely if already running from the cargo
    //    bin path (self_replace already updated it in-place). Non-fatal: if the
    //    copy fails, the running process is already updated via self_replace.
    let spawn_path = resolve_spawn_path(
        &binary_path,
        cargo_bin_path.as_deref(),
        admin_target.as_deref(),
    )
    .await?;

    // 8. Notify: build complete, restarting.
    notify_admin("✅ Build complete. Restarting…", admin_target.as_deref()).await;

    // 9. Notify: starting new instance (MUST be before step 10 shutdown —
    //    Telegram channel must still be live for this notification).
    notify_admin("🔄 Starting new instance…", admin_target.as_deref()).await;

    // Resolve current_exe() now so the post-spawn cleanup closure is
    // infallible (the original step-14 `?` propagated before the drain; with
    // the shared tail the closure must not fail after the checkpoint).
    let current_exe_path = std::env::current_exe().context("Failed to resolve current_exe()")?;

    // 10–15. Shared finalize tail: graceful drain, checkpoint, lock release,
    // spawn, build-artifact cleanup, exit. The artifact cleanup runs only after
    // a successful spawn (Gatekeeper safety), then the process exits.
    finalize_update_and_restart(&spawn_path, admin_target.as_deref(), || {
        // 14. Clean up the build output binary after successful spawn.
        //     Note: never delete:
        //     - The spawn target (prevents macOS Gatekeeper race — see step 13).
        //     - The current_exe path (same Gatekeeper concern).
        //     - The cargo bin path (same Gatekeeper concern, also the spawn target).
        //     All comparisons use canonicalized paths to handle symlinks correctly.
        let should_delete = should_delete_build_artifact(
            &binary_path,
            &current_exe_path,
            cargo_bin_path.as_deref(),
        );

        if should_delete {
            if let Err(e) = fs::remove_file(&binary_path) {
                warn!(
                    error = %e,
                    path = %binary_path.display(),
                    "Could not remove build artifact after successful spawn"
                );
            }
        } else {
            info!(
                path = %binary_path.display(),
                "Skipping deletion of build artifact (matches current_exe or cargo bin path)"
            );
        }
    })
    .await
}

/// Registry self-update: `cargo install <crate> --force` from crates.io,
/// then restart from the freshly installed binary.
///
/// See [`execute_update`] for the concurrent-guard and exit contracts.
async fn execute_registry_update() -> Result<()> {
    // Concurrent guard — only one update at a time.
    let Some(_guard) = UPDATE_MUTEX.try_lock().ok() else {
        anyhow::bail!("An update is already in progress. Please wait for it to complete.");
    };

    // Registry self-update is not offered on Windows (the running .exe cannot
    // be replaced). The GUI never shows the button there, but guard the entry
    // point as well so a stray call cannot half-update.
    if cfg!(windows) {
        anyhow::bail!("Registry self-update is not supported on Windows");
    }

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

    // 3. Run `cargo install <crate> --force`. No `--locked`: the published
    //    .crate ships a Cargo.lock, but a stale lockfile would hard-fail the
    //    install; plain `cargo install` (per the ticket) re-resolves when the
    //    lock is stale. 60-minute timeout — a cold registry build recompiles
    //    the whole dependency tree (no reuse of a local target/). The cwd is
    //    the storage root: `cargo install <crate>` needs no manifest dir, but
    //    a writable CWD — the daemon's launch directory may be anything (or
    //    removed).
    let crate_name = env!("CARGO_PKG_NAME");
    let install_cwd = crate::config::CONFIG.global_storage_root();
    run_cargo_with_timeout(
        &["install", crate_name, "--force"],
        &install_cwd,
        Duration::from_hours(1),
        &format!("cargo install {crate_name} --force"),
        "Update",
        admin_target.as_deref(),
    )
    .await?;

    // 4. Resolve the restart target: the binary `cargo install` just wrote.
    //    Prefer `current_exe()` when the install overwrote it in place;
    //    otherwise the cargo bin path holds the fresh copy.
    let spawn_path = resolve_registry_spawn_path(admin_target.as_deref()).await?;

    // 5. Notify: install complete, restarting.
    notify_admin(
        "✅ Update installed from crates.io. Restarting…",
        admin_target.as_deref(),
    )
    .await;

    // 6. Notify: starting new instance (MUST be before step 7 shutdown —
    //    Telegram channel must still be live for this notification).
    notify_admin("🔄 Starting new instance…", admin_target.as_deref()).await;

    // 7–12. Shared finalize tail: graceful drain, checkpoint, lock release,
    // spawn, exit. No build artifact to clean up in registry mode.
    finalize_update_and_restart(&spawn_path, admin_target.as_deref(), || {}).await
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
///    (or force-cancels at the 10-min cap). The GUI stays open with a
///    "shutting down…" state; the GUI exit path waits
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
/// 5. Run `after_spawn` (local mode: delete the build artifact; registry mode:
///    no-op), then `exit(0)`.
///
/// `after_spawn` must not fail the update (best-effort cleanup) — a panic here
/// would still exit the process, so it must be infallible in practice.
async fn finalize_update_and_restart(
    spawn_path: &Path,
    admin_target: Option<&str>,
    after_spawn: impl FnOnce(),
) -> Result<()> {
    // 10. Begin the graceful drain (local numbering; shared across modes).
    UPDATE_FINALIZING.store(true, std::sync::atomic::Ordering::SeqCst);
    crate::shutdown::drain_begin();
    while !crate::shutdown::shutdown_token().is_cancelled() {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    crate::tools::browser::close_all_browser_sessions().await;

    // 11. Checkpoint all databases BEFORE releasing the instance lock and
    //     spawning the replacement (see doc comment above).
    crate::checkpoint::checkpoint_all_databases().await;

    // 12. Release instance lock so the child process can acquire it on startup.
    release_instance_lock().await;

    // 13. Spawn the new instance from the determined spawn path.
    if let Err(e) = spawn_new_instance_from(spawn_path, admin_target).await {
        // Spawn failed — the process stays alive (unless a genuine window
        // close was requested during the finalizing window, in which case the
        // GUI honors it with its own checkpoint + exit via UpdateResult).
        // Clear the finalizing flag and re-acquire the lock.
        UPDATE_FINALIZING.store(false, std::sync::atomic::Ordering::SeqCst);
        // Re-acquire the lock since the process stays alive.
        if let Err(lock_err) = reacquire_instance_lock().await {
            error!(%lock_err, "Failed to re-acquire instance lock after spawn failure");
        }
        return Err(e);
    }

    // 14. Post-spawn cleanup (local mode: delete build artifact).
    after_spawn();

    // 15. Exit — spawn succeeded.
    std::process::exit(0);
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Run a long-running cargo subcommand with a timeout, notifying the admin
/// on failure. Shared by the local-build and registry-install update paths
/// (the only differences are the arguments, working directory, timeout, and
/// user-facing labels).
///
/// `args` are the cargo arguments (excluding the `cargo` binary itself),
/// `cwd` the working directory, `timeout` the hard deadline, `label` a
/// short human-readable description used in logs and error messages (e.g.
/// "cargo build --release"), and `failure_kind` the noun used in failure
/// messages ("Build" / "Update").
async fn run_cargo_with_timeout(
    args: &[&str],
    cwd: &Path,
    timeout: Duration,
    label: &str,
    failure_kind: &str,
    admin_target: Option<&str>,
) -> Result<()> {
    info!("Starting {label} in {}", cwd.display());
    let cargo_result = tokio::time::timeout(
        timeout,
        tokio::process::Command::new("cargo")
            .args(args)
            .current_dir(cwd)
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
            let msg = format!(
                "❌ {failure_kind} failed: {label} timed out after {} minutes",
                timeout.as_secs() / 60
            );
            notify_admin(&msg, admin_target).await;
            anyhow::bail!(msg);
        }
        Ok(Err(e)) => {
            let msg = format!("❌ {failure_kind} failed: could not start cargo: {e}");
            notify_admin(&msg, admin_target).await;
            anyhow::bail!(msg);
        }
        Ok(Ok(output)) if !output.status.success() => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let combined = format!("stdout:\n{stdout}\nstderr:\n{stderr}");
            let truncated = truncate_to_last_64k(&combined);
            let msg = format!("❌ {failure_kind} failed:\n```\n{truncated}\n```");
            notify_admin(&msg, admin_target).await;
            anyhow::bail!("{label} failed with exit status: {}", output.status);
        }
        Ok(Ok(_)) => {
            info!("{label} completed successfully");
            Ok(())
        }
    }
}

/// Resolve the restart target after a successful `cargo install --force`.
///
/// The freshly installed binary is the correct spawn target. `cargo install`
/// honors `install.root` config / `--root`, so the fresh copy may overwrite
/// `current_exe()` in place OR land at the default cargo bin path — compare
/// modification times and spawn the newest executable. When the freshly
/// installed binary is not the running one, the admin is notified that the
/// old copy is left stale.
///
/// **Known limitation (custom install roots):** when `install.root` /
/// `CARGO_INSTALL_ROOT` / `--root` points somewhere that is neither
/// `current_exe()` nor the default cargo bin path (`$CARGO_HOME/bin` /
/// `~/.cargo/bin`), the fresh binary lands at a path this function never
/// looks at, so it falls back to `current_exe()` — no admin notification
/// fires (the "fresh elsewhere" branch below is not taken) and the update
/// appears to succeed while the PATH-visible binary stays stale, so the next
/// manual start still runs the old version. Detecting the true install root
/// would require replicating cargo's full config resolution, so this narrow
/// case is accepted and documented rather than guessed.
///
/// The chosen spawn target is validated with [`is_executable`] before being
/// returned (mirroring [`resolve_spawn_path`]); if it is not executable, falls
/// back to `current_exe()`.
async fn resolve_registry_spawn_path(admin_target: Option<&str>) -> Result<PathBuf> {
    let current_exe = std::env::current_exe()
        .context("Failed to resolve current_exe() for registry update restart")?;

    // `cargo install --force` overwrote the running binary in place — restart
    // from current_exe (the common default-root case).
    let cargo_bin = resolve_cargo_bin_path();
    if let Some(cargo_bin) = &cargo_bin
        && canonicalize_safe(&current_exe) == canonicalize_safe(cargo_bin)
    {
        info!(
            "cargo install updated the running binary in place at `{}` — restarting from current_exe",
            current_exe.display()
        );
        return Ok(current_exe);
    }

    // The fresh copy landed elsewhere (custom `--root`, `install.root` config,
    // or a manually-moved binary). Prefer whichever of `current_exe()` and the
    // cargo bin path is newest — the just-installed binary has the newest
    // mtime. Notify the admin when the running copy is left stale.
    let mtime =
        |path: &Path| -> Option<std::time::SystemTime> { fs::metadata(path).ok()?.modified().ok() };
    let current_mtime = mtime(&current_exe);
    let mut spawn = current_exe.clone();
    let mut fresh_elsewhere = false;

    if let Some(cargo_bin) = &cargo_bin
        && is_executable(cargo_bin)
    {
        let cargo_mtime = mtime(cargo_bin);
        let cargo_fresher = match (cargo_mtime, current_mtime) {
            (Some(c), Some(s)) => c > s,
            (Some(_), None) => true,
            (None, _) => false,
        };
        if cargo_fresher {
            spawn = cargo_bin.clone();
            fresh_elsewhere = true;
        }
    }

    // Validate the chosen spawn target exists and is executable.
    if !is_executable(&spawn) {
        warn!(
            path = %spawn.display(),
            "Registry spawn target not executable — falling back to current_exe()"
        );
        if !is_executable(&current_exe) {
            anyhow::bail!(
                "Neither cargo bin path `{}` nor current_exe `{}` is executable",
                spawn.display(),
                current_exe.display(),
            );
        }
        return Ok(current_exe);
    }

    if fresh_elsewhere {
        notify_admin(
            &format!(
                "⚠️ Update installed to `{}`. The previously running copy at `{}` \
                 was not updated in place (different install root). \
                 Restarting from the freshly installed binary.",
                spawn.display(),
                current_exe.display(),
            ),
            admin_target,
        )
        .await;
        info!(
            path = %spawn.display(),
            "Restarting from freshly installed binary (running copy was stale)"
        );
    } else {
        // No fresher binary at the cargo bin path: either the install
        // overwrote `current_exe()` in place (canonical-equality check
        // missed only if cargo bin resolution failed) or the install landed
        // at a custom root this function cannot see (documented limitation).
        info!(
            path = %spawn.display(),
            "Restarting from current_exe (no fresher binary at the cargo bin path)"
        );
    }
    Ok(spawn)
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

// ── Cargo bin path resolution and installation ───────────────────────────

/// Resolve the path to the `mahbot` binary in the cargo bin directory.
///
/// Delegates to [`crate::util::cargo_bin_dir`] for directory resolution,
/// then appends the platform-specific executable name.
fn resolve_cargo_bin_path() -> Option<PathBuf> {
    let exe_name = format!("mahbot{}", std::env::consts::EXE_SUFFIX);
    Some(crate::util::cargo_bin_dir()?.join(exe_name))
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
/// updated via `self_replace`. On failure, logs a warning, attempts admin
/// notification, and returns `None` to signal the caller to fall back to
/// `current_exe()` for spawning.
async fn copy_to_cargo_bin(
    source: &Path,
    dest: &Path,
    admin_target: Option<&str>,
) -> Option<PathBuf> {
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
        return None;
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
        return None;
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
        return None;
    }

    info!(path = %dest.display(), "Installed new binary to cargo bin path");
    Some(dest.to_path_buf())
}

/// Determine whether the build artifact at `binary_path` can be safely deleted.
///
/// Returns `true` only when `binary_path` differs from both the current
/// executable path and the cargo install path (after canonicalization).
/// This guarantees the spawn target is never deleted, preventing the macOS
/// Gatekeeper race (see [`execute_update`] for details).
fn should_delete_build_artifact(
    binary_path: &Path,
    current_exe_path: &Path,
    cargo_bin_path: Option<&Path>,
) -> bool {
    let binary_canon = canonicalize_safe(binary_path);
    let current_exe_canon = canonicalize_safe(current_exe_path);
    let cargo_bin_canon = cargo_bin_path.map(canonicalize_safe);

    binary_canon != current_exe_canon && (cargo_bin_canon.as_ref() != Some(&binary_canon))
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

/// Determine the spawn target path after a successful build and self_replace.
///
/// Returns the cargo bin install path if the copy succeeds, or falls back to
/// `current_exe()` if:
/// - No cargo bin path could be resolved (no `CARGO_HOME` or `UserDirs`).
/// - The binary is already running from the cargo bin path (self_replace
///   already updated it in-place).
/// - The copy to the cargo bin path fails.
///
/// Validates that the chosen spawn target exists and is executable. If
/// validation fails, falls back to `current_exe()`.
async fn resolve_spawn_path(
    built_binary: &Path,
    cargo_bin: Option<&Path>,
    admin_target: Option<&str>,
) -> Result<PathBuf> {
    let current_exe = std::env::current_exe()
        .context("Failed to resolve current_exe() for spawn path resolution")?;

    let candidate = if let Some(cargo_bin) = cargo_bin {
        // If we're already running from the cargo bin path, self_replace
        // already updated it in-place — skip the copy.
        if canonicalize_safe(&current_exe) == canonicalize_safe(cargo_bin) {
            info!(
                "Already running from cargo bin path `{}` — skipping install copy",
                cargo_bin.display()
            );
            cargo_bin.to_path_buf()
        } else {
            // Attempt the install copy; fall back to current_exe() on failure.
            copy_to_cargo_bin(built_binary, cargo_bin, admin_target)
                .await
                .unwrap_or_else(|| current_exe.clone())
        }
    } else {
        // No cargo bin path could be resolved (no $CARGO_HOME, no home dir).
        current_exe.clone()
    };

    // Validate the chosen spawn target exists and is executable.
    if !is_executable(&candidate) {
        warn!(
            path = %candidate.display(),
            "Primary spawn target not executable — falling back to current_exe()"
        );
        if !is_executable(&current_exe) {
            anyhow::bail!(
                "Neither cargo bin path `{}` nor current_exe `{}` is executable",
                candidate.display(),
                current_exe.display(),
            );
        }
        return Ok(current_exe);
    }

    Ok(candidate)
}

/// Spawn the new mahbot instance as a detached child process from the given path.
///
/// The `binary_path` must point to an existing, executable binary (typically the
/// cargo install bin path or `current_exe()` as fallback).
///
/// On Unix: null stdin/stdout, stderr → update.log. On Windows: same + `DETACHED_PROCESS | CREATE_NO_WINDOW`.
/// On spawn failure: notifies admin, keeps running (does NOT exit).
///
/// ## macOS Gatekeeper safety
///
/// The caller guarantees that `binary_path` is never deleted before or during
/// the child's startup window (see deletion safety in [`execute_update`]).
/// Deleting the spawn target while Gatekeeper is validating its code signature
/// causes `syspolicyd` to SIGKILL the child.
async fn spawn_new_instance_from(binary_path: &Path, admin_target: Option<&str>) -> Result<()> {
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
            let msg = format!("❌ Failed to start new instance: {e}");
            notify_admin(&msg, admin_target).await;
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
    use crate::lock_utils::{lock_file_path, try_flock};

    /// Make a file executable (0o755) on Unix; no-op on other platforms.
    fn make_executable(path: &Path) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, PermissionsExt::from_mode(0o755)).unwrap();
        }
        #[cfg(not(unix))]
        let _ = path;
    }

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
    fn test_is_update_available() {
        // This test runs from the MahBot repo — a local checkout (`.git`
        // present) with a reachable Cargo.toml — so self-update is available.
        assert!(
            is_update_available(),
            "Self-update should be available when running from repo"
        );
        assert_eq!(
            update_mode(),
            UpdateMode::LocalCheckout,
            "The repo checkout must classify as LocalCheckout"
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

    #[cfg(unix)]
    #[test]
    fn test_is_executable_on_unix() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test_exe");

        // File doesn't exist — should not be executable.
        assert!(!is_executable(&file_path));

        // Create a non-executable file.
        std::fs::write(&file_path, "content").unwrap();
        std::fs::set_permissions(&file_path, PermissionsExt::from_mode(0o644)).unwrap();
        assert!(
            !is_executable(&file_path),
            "File with mode 644 should not be executable"
        );

        // Set executable bit.
        make_executable(&file_path);
        assert!(
            is_executable(&file_path),
            "File with mode 755 should be executable"
        );

        // Also test with only owner execute bit.
        std::fs::set_permissions(&file_path, PermissionsExt::from_mode(0o100)).unwrap();
        assert!(
            is_executable(&file_path),
            "File with mode 100 should be executable"
        );
    }

    #[cfg(windows)]
    #[test]
    fn test_is_executable_on_windows() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test_exe.exe");

        // File doesn't exist — should not be executable.
        assert!(!is_executable(&file_path));

        // Create an exe file.
        std::fs::write(&file_path, "content").unwrap();
        assert!(
            is_executable(&file_path),
            "File with .exe extension should be executable"
        );

        // Non-exe file should not be executable.
        let txt_path = dir.path().join("test.txt");
        std::fs::write(&txt_path, "content").unwrap();
        assert!(
            !is_executable(&txt_path),
            "File with .txt extension should not be executable"
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

        // Copy should succeed.
        let result = copy_to_cargo_bin(&source, &dest, None).await;
        assert!(result.is_some(), "Copy should succeed");
        assert_eq!(result.unwrap(), dest);

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

        // Copy should fail gracefully.
        let result = copy_to_cargo_bin(&source, &dest, None).await;
        assert!(result.is_none(), "Copy should return None on failure");
        assert!(!dest.exists(), "Destination should not be created");
    }

    #[tokio::test]
    async fn test_copy_to_cargo_bin_creates_parent_dir() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source_bin");
        let dest = dir.path().join("deep").join("nested").join("installed_bin");

        std::fs::write(&source, "content").unwrap();

        // Copy should create parent directories.
        let result = copy_to_cargo_bin(&source, &dest, None).await;
        assert!(
            result.is_some(),
            "Copy should create parent dirs and succeed"
        );
        assert!(dest.is_file(), "Destination should exist");
        assert!(
            dest.parent().unwrap().is_dir(),
            "Parent directory should exist"
        );
    }

    #[test]
    fn test_should_delete_build_artifact() {
        let cases: &[(&str, &str, Option<&str>, bool)] = &[
            (
                "/usr/local/bin/mahbot",
                "/usr/local/bin/mahbot",
                Some("/usr/local/bin/mahbot"),
                false,
            ), // all same
            (
                "/build/target/release/mahbot",
                "/usr/local/bin/mahbot",
                Some("/home/user/.cargo/bin/mahbot"),
                true,
            ), // all different
            (
                "/home/user/.cargo/bin/mahbot",
                "/home/user/dev/mahbot/target/release/mahbot",
                Some("/home/user/.cargo/bin/mahbot"),
                false,
            ), // binary matches cargo
            (
                "/usr/local/bin/mahbot",
                "/usr/local/bin/mahbot",
                Some("/home/user/.cargo/bin/mahbot"),
                false,
            ), // binary matches current, differs from cargo
            (
                "/build/target/release/mahbot",
                "/usr/local/bin/mahbot",
                None,
                true,
            ), // no cargo bin, differs
            (
                "/usr/local/bin/mahbot",
                "/usr/local/bin/mahbot",
                None,
                false,
            ), // no cargo bin, same
        ];
        for &(binary, current, cargo_bin, expected) in cases {
            assert_eq!(
                should_delete_build_artifact(
                    Path::new(binary),
                    Path::new(current),
                    cargo_bin.map(Path::new)
                ),
                expected,
                "binary={binary}, current={current}, cargo_bin={cargo_bin:?}"
            );
        }
    }

    #[tokio::test]
    async fn test_resolve_spawn_path_falls_back_to_current_exe_on_copy_failure() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("nonexistent_source"); // doesn't exist
        let dest = dir.path().join("install").join("mahbot");
        let current_exe = std::env::current_exe().unwrap();

        let result = resolve_spawn_path(&source, Some(dest.as_path()), None).await;

        assert!(
            result.is_ok(),
            "Should fall back to current_exe on copy failure"
        );
        assert_eq!(
            result.unwrap(),
            current_exe,
            "Should return current_exe when copy fails"
        );
    }

    #[tokio::test]
    async fn test_resolve_spawn_path_no_cargo_bin() {
        let source = Path::new("/tmp/nonexistent_binary");
        let current_exe = std::env::current_exe().unwrap();

        let result = resolve_spawn_path(source, None, None).await;

        assert!(
            result.is_ok(),
            "Should return current_exe when no cargo bin path"
        );
        assert_eq!(result.unwrap(), current_exe);
    }

    #[tokio::test]
    async fn test_resolve_spawn_path_copy_success() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("built_bin");
        let dest = dir.path().join("cargo_bin").join("mahbot");

        // Create an executable source binary.
        std::fs::write(&source, "binary payload").unwrap();
        make_executable(&source);

        // On success, resolve_spawn_path should return the cargo bin path
        // (the dest path), not current_exe().
        let result = resolve_spawn_path(&source, Some(dest.as_path()), None).await;
        assert!(result.is_ok(), "resolve_spawn_path should succeed");
        let path = result.unwrap();
        assert_eq!(
            path, dest,
            "Should return the cargo bin path on successful copy"
        );
        assert!(dest.is_file(), "Destination should exist");
        assert_eq!(std::fs::read_to_string(&dest).unwrap(), "binary payload");
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
