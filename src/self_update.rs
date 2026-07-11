//! Self-update logic — single-instance guarding, build, binary swap, and restart.
//!
//! Uses `flock()` for single-instance enforcement (kernel guarantees lock release on
//! process death). The update flow: `cargo build --release` → `self_replace` →
//! copy to cargo install bin → release lock → spawn new instance from cargo bin
//! path (or `current_exe()` fallback) → remove build artifact (guarded against
//! deleting the spawn target, `current_exe()`, or the cargo bin path) →
//! checkpoint all Turso databases → `exit(0)`.
//!
//! The cargo install path resolution uses `$CARGO_HOME` if set, else
//! `~/.cargo/bin` via `directories::UserDirs` — this ensures the
//! self-updated binary is visible to the shell tool and the user's PATH
//! (the shell tool's `extra_shell_path_prefixes` includes both paths when
//! they differ, so the single resolved path is always covered).
//!
//! The WAL checkpoint before `exit(0)` is critical: `std::process::exit(0)` bypasses
//! all Rust destructors, so Turso connections are never properly closed. Without an
//! explicit checkpoint, pending WAL writes are silently lost and `.tshm` coordination
//! files are left inconsistent, causing data to reappear after restart.
//!
//! ## macOS Gatekeeper safety
//!
//! `posix_spawn` triggers async Gatekeeper code-signing validation; deleting the
//! spawn target during validation produces empty stderr (SIGKILL by `syspolicyd`).
//! See `should_delete_build_artifact()` and `execute_update()` steps 12–13.
//!
//! Self-update is only available when running from the original build checkout
//! directory — `CARGO_MANIFEST_DIR` is a compile-time constant embedded via
//! `env!("CARGO_MANIFEST_DIR")`.

use anyhow::{Context, Result, anyhow};
#[cfg(test)]
use directories::UserDirs;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{LazyLock, OnceLock};
use tokio::sync::Mutex;
use tracing::{error, info, warn};

// ── File-lock based single-instance guard ─────────────────────────────────

/// Acquire an exclusive lock on the global lock file, failing immediately
/// if another instance holds it. Retries up to 3 times with 100ms delays
/// to handle the scheduling window between old process exit and kernel lock
/// release.
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
    let lock_path = lock_file_path(storage_root);

    // Ensure parent directory exists.
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }

    match try_acquire_lock(&lock_path)? {
        Some(mut file) => {
            // Write our PID to the lock file for diagnostics only (not used
            // for alive-checking — the lock itself is authoritative).
            let _ = write!(file, "{}", std::process::id());
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
/// Retries up to 3 times with 100ms delays between attempts, to handle the
/// scheduling window between an old process exiting and the kernel releasing
/// its lock.
///
/// # Returns
///
/// - `Ok(Some(file))` — lock acquired successfully. **The caller must keep the
///   returned `File` alive for the lifetime of the lock** — dropping it releases
///   the kernel-level lock.
/// - `Ok(None)` — all retries exhausted (another process holds the lock).
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
/// - **PID writing & error messages**: each caller formats its own success/failure
///   messages.
fn try_acquire_lock(path: &Path) -> Result<Option<File>> {
    for attempt in 0..3 {
        let file = open_lock_file(path)
            .with_context(|| format!("failed to open lock file {}", path.display()))?;

        if try_flock(&file)? {
            return Ok(Some(file));
        }

        if attempt < 2 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    Ok(None)
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

/// Path to the global lock file under the given storage root.
fn lock_file_path(storage_root: &Path) -> PathBuf {
    storage_root.join("mahbot.lock")
}

#[cfg(unix)]
fn try_flock(file: &File) -> Result<bool> {
    use std::os::unix::io::AsRawFd;
    let fd = file.as_raw_fd();

    let result = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    if result == 0 {
        Ok(true)
    } else {
        let err = std::io::Error::last_os_error();
        match err.raw_os_error() {
            Some(libc::EAGAIN) => Ok(false), // EAGAIN == EWOULDBLOCK on most platforms
            _ => Err(anyhow::Error::from(err).context("flock failed on lock file")),
        }
    }
}

#[cfg(windows)]
fn try_flock(file: &File) -> Result<bool> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{ERROR_LOCK_VIOLATION, HANDLE};
    use windows_sys::Win32::Storage::FileSystem::{
        LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY, LockFileEx,
    };

    let handle = file.as_raw_handle() as HANDLE;

    const LOCK_VIOLATION: i32 = ERROR_LOCK_VIOLATION as i32;

    let mut overlapped =
        unsafe { std::mem::zeroed::<windows_sys::Win32::System::IO::OVERLAPPED>() };
    let locked = unsafe {
        LockFileEx(
            handle,
            LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
            0,
            0,
            0,
            &mut overlapped,
        )
    };

    if locked != 0 {
        Ok(true)
    } else {
        let err = std::io::Error::last_os_error();
        match err.raw_os_error() {
            Some(LOCK_VIOLATION) => Ok(false),
            _ => Err(anyhow::Error::from(err).context("LockFileEx failed on lock file")),
        }
    }
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
/// Uses [`tokio::task::spawn_blocking`] to offload the blocking retry loop
/// (which uses `std::thread::sleep`) to a dedicated blocking thread, avoiding
/// blocking the Tokio worker thread.
///
/// This is a recoverable path: it runs during self-update after all agents
/// have been cancelled, browser sessions closed, and shutdown signaled.
async fn reacquire_instance_lock() -> Result<()> {
    let mutex = INSTANCE_LOCK
        .get()
        .context("Instance lock not initialized")?;

    // Extract the lock path while the mutex is held, dropping the guard
    // before spawn_blocking to avoid holding the tokio Mutex across the
    // blocking thread boundary.
    let lock_path = {
        let guard = mutex.lock().await;
        if guard.file.is_some() {
            return Ok(()); // Already held.
        }
        guard.lock_path.clone()
    };

    // Offload blocking retry loop (std::thread::sleep) to a blocking thread.
    let file = tokio::task::spawn_blocking(move || try_acquire_lock(&lock_path))
        .await
        .context("spawn_blocking for lock reacquire failed")??;

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

/// Check whether self-update is available.
///
/// Self-update requires running from the original build checkout directory,
/// which is detected by the presence of `{CARGO_MANIFEST_DIR}/Cargo.toml`.
/// `CARGO_MANIFEST_DIR` is a compile-time constant — if the binary was moved
/// from its build directory, this returns `false`.
#[must_use]
pub fn is_update_available() -> bool {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("Cargo.toml")
        .is_file()
}

// ── Update mutex ──────────────────────────────────────────────────────────

/// Global mutex ensuring only one update runs at a time.
/// A second trigger while an update is in progress gets an immediate error
/// via [`try_lock`](Mutex::try_lock).
static UPDATE_MUTEX: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

// ── Execute update ────────────────────────────────────────────────────────

/// Execute a self-update: build from source, swap binary, install to cargo bin,
/// notify admin, restart.
///
/// Called from the GUI update button.
/// Only one update runs at a time — concurrent calls return an error immediately.
///
/// On success, this function never returns (`std::process::exit(0)`).
/// On failure, returns an error.
pub async fn execute_update() -> Result<()> {
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
    match tokio::process::Command::new("cargo")
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
    {
        Ok(status) if status.success() => {}
        _ => anyhow::bail!("cargo not found on PATH — cannot build from source"),
    }

    // 2. Look up admin Telegram reply_target.
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
    run_cargo_build(manifest_dir, admin_target.as_deref()).await?;

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

    // 10. Shutdown: cancel all agents, close browser sessions, signal shutdown.
    crate::registry::AGENT_REGISTRY.shutdown_all();
    crate::tools::browser::close_all_browser_sessions().await;
    crate::shutdown::shutdown();

    // 11. Release instance lock so the child process can acquire it on startup.
    release_instance_lock().await;

    // 12. Spawn the new instance from the determined spawn path.
    //     On macOS, posix_spawn triggers asynchronous Gatekeeper code signature
    //     validation. Spawning before deletion guarantees the spawn target is
    //     never deleted during or before the child's startup window.
    if let Err(e) = spawn_new_instance_from(&spawn_path, admin_target.as_deref()).await {
        // Spawn failed — re-acquire the lock since the process stays alive.
        if let Err(lock_err) = reacquire_instance_lock().await {
            error!(%lock_err, "Failed to re-acquire instance lock after spawn failure");
        }
        return Err(e);
    }

    // 13. Clean up the build output binary after successful spawn.
    //     Note: never delete:
    //     - The spawn target (prevents macOS Gatekeeper race — see step 12).
    //     - The current_exe path (same Gatekeeper concern).
    //     - The cargo bin path (same Gatekeeper concern, also the spawn target).
    //     All comparisons use canonicalized paths to handle symlinks correctly.
    let current_exe_path = std::env::current_exe().context("Failed to resolve current_exe()")?;
    let should_delete =
        should_delete_build_artifact(&binary_path, &current_exe_path, cargo_bin_path.as_deref());

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

    // 14. Checkpoint all databases to prevent WAL data loss.
    //     `exit(0)` below bypasses Rust destructors, so Turso connections are
    //     never properly closed. Without this checkpoint, pending WAL writes are
    //     silently lost (e.g., archived tickets reappear after restart).
    crate::checkpoint::checkpoint_all_databases().await;

    // 15. Exit — spawn succeeded.
    std::process::exit(0);
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Run `cargo build --release` with a 30-minute timeout.
/// On failure, notifies admin and returns an error.
async fn run_cargo_build(manifest_dir: &Path, admin_target: Option<&str>) -> Result<()> {
    info!(
        "Starting cargo build --release in {}",
        manifest_dir.display()
    );
    let build_result = tokio::time::timeout(
        std::time::Duration::from_mins(30),
        tokio::process::Command::new("cargo")
            .args(["build", "--release", "--locked"])
            .current_dir(manifest_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output(),
    )
    .await;

    match build_result {
        Err(_elapsed) => {
            let msg = "❌ Build failed: timed out after 30 minutes";
            notify_admin(msg, admin_target).await;
            anyhow::bail!(msg);
        }
        Ok(Err(e)) => {
            let msg = format!("❌ Build failed: could not start cargo: {e}");
            notify_admin(&msg, admin_target).await;
            anyhow::bail!(msg);
        }
        Ok(Ok(output)) if !output.status.success() => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let combined = format!("stdout:\n{stdout}\nstderr:\n{stderr}");
            let truncated = truncate_to_last_64k(&combined);
            let msg = format!("❌ Build failed:\n```\n{truncated}\n```");
            notify_admin(&msg, admin_target).await;
            anyhow::bail!("Build failed with exit status: {}", output.status);
        }
        Ok(Ok(_)) => {
            info!("cargo build --release completed successfully");
            Ok(())
        }
    }
}

/// Look up the admin user's Telegram reply target.
///
/// Returns `Some(reply_target)` if the "admin" user has a Telegram channel
/// binding with a non-null `reply_target` and a bot token is configured.
/// Returns `None` otherwise.
pub async fn resolve_admin_telegram_target() -> Option<String> {
    let _ = crate::config::CONFIG.telegram_bot_token()?;

    let store = crate::users::store();
    let bindings = store.get_user_channels("admin").await.ok()?;
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

    let Some(channel) = crate::channel_registry().get("telegram") else {
        warn!("Telegram channel not found in registry — cannot send update notification");
        return;
    };

    let reply = crate::SendMessage {
        content: message.to_string(),
        recipient: recipient.to_string(),
        reply_markup: None,
    };

    if let Err(e) = channel.send(&reply).await {
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

/// Check whether a path points to an executable file.
///
/// On Unix: checks that the file exists and has at least one execute bit set
/// (owner, group, or other). Uses `PermissionsExt::mode() & 0o111`.
///
/// On Windows: checks that the file exists and has a `.exe` extension.
/// (Windows executability is determined by extension and content, not
/// permission bits.)
#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.is_file() && fs::metadata(path).is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
}

#[cfg(windows)]
fn is_executable(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .map_or(false, |ext| ext.eq_ignore_ascii_case("exe"))
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
    fn test_try_acquire_lock_exhaustion() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = dir.path().join("mahbot.lock");

        // Hold the lock on this file so try_acquire_lock will fail.
        let holder = open_lock_file(&lock_path).unwrap();
        assert!(try_flock(&holder).unwrap(), "First flock should succeed");

        // While the lock is held, try_acquire_lock should exhaust retries.
        let result = try_acquire_lock(&lock_path).unwrap();
        assert!(result.is_none(), "Should exhaust retries when lock is held");

        // Release the lock.
        drop(holder);

        // After release, try_acquire_lock should succeed.
        let result = try_acquire_lock(&lock_path).unwrap();
        assert!(
            result.is_some(),
            "Should acquire lock after previous holder releases"
        );
    }

    #[test]
    fn test_is_update_available() {
        // This test runs from the MahBot repo, so it should be available.
        assert!(
            is_update_available(),
            "Self-update should be available when running from repo"
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
