//! Private daemon temp root: one `/tmp/mahbot-<uid>` root for ALL daemon temp
//! artifacts, plus the boot sweep and the one-time legacy temp-dir cleanup.
//!
//! ## One root
//!
//! The daemon creates a single private root under `/tmp` (`/tmp/mahbot-<uid>`,
//! mode 0700) with exclusive-create + ownership/mode verification, and fails
//! loudly on a squatted path. It then pins its own `TMPDIR` to that root at
//! the very start of startup (before config and any temp use), so every
//! `std::env::temp_dir()`-based consumer relocates automatically: shell spill
//! files, research run folders, background-mode output, voice/telegram temp.
//! The OS removes empty directories on its 3-day temp sweep, so the root is
//! recreated and re-verified on every boot.
//!
//! Shell children get `TMPDIR` set to the same root (see
//! [`crate::tools::shell::shell_tmpdir`]).
//!
//! ## Boot sweep + legacy cleanup
//!
//! At real daemon boot [`boot_sweep`] sweeps crash leftovers in the root
//! (`.agent` spill files + liveness-guarded run-root sweep) and cleans the
//! mahbot-managed bases left by previous generations in the pre-pin legacy
//! temp dir: `.agent`, `mahbot_voice`, `mahbot_telegram_files` exactly ONCE
//! (marker-gated), plus the legacy `mahbot-research` base EVERY boot
//! (liveness-guarded so live pre-upgrade run folders are preserved — and
//! folders preserved on an earlier boot are reclaimed once their runs
//! terminalize, instead of orphaning forever).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The pinned private temp root, set once by [`init_temp_root`].
static TEMP_ROOT: OnceLock<PathBuf> = OnceLock::new();

/// The pre-pin OS temp dir (e.g. `/var/folders/.../T` on macOS). Captured
/// before the `TMPDIR` pin so the readonly guard keeps it as an allowed root
/// (bare macOS `mktemp` ignores `TMPDIR` and lands there) and the one-time
/// legacy cleanup can target it.
static LEGACY_TEMP_DIR: OnceLock<PathBuf> = OnceLock::new();

/// The pinned root path, if [`init_temp_root`] ran (production unix startup).
#[must_use]
pub(crate) fn temp_root() -> Option<&'static Path> {
    TEMP_ROOT.get().map(PathBuf::as_path)
}

/// The pre-pin OS temp dir (the legacy darwin dir on macOS), if captured.
#[must_use]
pub(crate) fn legacy_temp_dir() -> Option<&'static Path> {
    LEGACY_TEMP_DIR.get().map(PathBuf::as_path)
}

/// Initialize the private temp root and pin `TMPDIR` to it.
///
/// Must run at the very start of startup, BEFORE config and any temp use, and
/// AFTER the debug/`__grep-engine` subcommand dispatches (those must not
/// create the root). Unix-only: `/tmp/mahbot-<uid>` is meaningless on Windows,
/// where the shell env uses `TEMP`/`TMP` instead.
///
/// Failure modes (fail loudly, never paper over):
/// - the root exists but is not a directory;
/// - the root exists and is owned by a different uid (squatting);
/// - the root's mode has group/other bits AND re-chmod fails — a loose mode on
///   OUR OWN path (uid verified) is self-healed to 0700 (a previous boot's
///   create-time chmod can fail on a race with the umask; bricking startup
///   forever over that would be worse).
#[cfg(unix)]
pub fn init_temp_root() -> anyhow::Result<()> {
    // Capture the legacy temp dir BEFORE the pin — on macOS this is the
    // darwin user temp dir (`/var/folders/.../T`) that bare `mktemp` uses.
    let legacy = std::env::temp_dir();
    let uid = unsafe { libc::geteuid() };
    let root = PathBuf::from(format!("/tmp/mahbot-{uid}"));

    // Exclusive create with explicit mode 0700. `create_dir` fails when the
    // path already exists (exclusive semantics); the mode is applied
    // explicitly because `create_dir` honors the process umask.
    match std::fs::create_dir(&root) {
        Ok(()) => {
            std::fs::set_permissions(&root, std::os::unix::fs::PermissionsExt::from_mode(0o700))
                .map_err(|e| {
                    anyhow::anyhow!("temp root {}: chmod 0700 failed: {e}", root.display())
                })?;
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            // Reuse after verification: ownership + mode + dir-ness.
            use std::os::unix::fs::MetadataExt;
            let meta = std::fs::symlink_metadata(&root)
                .map_err(|e| anyhow::anyhow!("temp root {}: stat failed: {e}", root.display()))?;
            if !meta.is_dir() {
                anyhow::bail!(
                    "temp root {} exists and is not a directory — refusing to use it",
                    root.display()
                );
            }
            if meta.uid() != uid {
                anyhow::bail!(
                    "temp root {} is owned by uid {} (expected {uid}) — refusing a squatted path",
                    root.display(),
                    meta.uid()
                );
            }
            let mode = meta.mode() & 0o777;
            if mode & 0o077 != 0 {
                // Self-heal: a loose mode on OUR OWN path is a previous
                // boot's create-time chmod failure (umask raced it), NOT a
                // squatter — the uid check above already proved ownership.
                // Re-chmod 0700 instead of bricking startup forever.
                std::fs::set_permissions(&root, std::os::unix::fs::PermissionsExt::from_mode(0o700))
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "temp root {} has group/other permissions (mode {mode:o}) and re-chmod 0700 failed: {e}",
                            root.display()
                        )
                    })?;
                tracing::warn!(
                    root = %root.display(),
                    mode = format_args!("{mode:o}"),
                    "Temp root had loose permissions — re-chmod 0700 (self-heal, path owned by self)"
                );
            }
        }
        Err(e) => anyhow::bail!("temp root {}: create failed: {e}", root.display()),
    }

    let _ = TEMP_ROOT.set(root.clone());
    let _ = LEGACY_TEMP_DIR.set(legacy);

    // Pin TMPDIR before any temp use. SAFETY: single-threaded startup (before
    // the tokio runtime / iced), no concurrent env access.
    unsafe { std::env::set_var("TMPDIR", &root) };
    tracing::info!(root = %root.display(), "Pinned daemon temp root");
    Ok(())
}

/// No-op on non-unix (the shell env there uses `TEMP`/`TMP`, not `TMPDIR`).
#[cfg(not(unix))]
pub fn init_temp_root() -> anyhow::Result<()> {
    Ok(())
}

/// The `TMPDIR` value for shell children: the pinned root when available,
/// otherwise the historical `"/tmp"` baseline (tests / non-unix).
#[must_use]
pub(crate) fn shell_tmpdir() -> String {
    temp_root().map_or_else(|| "/tmp".to_string(), |p| p.to_string_lossy().into_owned())
}

/// Where a BARE `mktemp -d` (no `-p`/template) actually lands on this
/// platform. On macOS bare mktemp ignores `TMPDIR` and uses
/// `_CS_DARWIN_USER_TEMP_DIR` (the legacy darwin dir); elsewhere mktemp
/// honors `TMPDIR`. The readonly guard's synthetic mktemp anchor must match
/// this, so `..` chains over it resolve like the real value.
#[must_use]
pub(crate) fn bare_mktemp_landing_root() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        if let Some(legacy) = legacy_temp_dir() {
            return legacy.to_path_buf();
        }
    }
    std::env::temp_dir()
}

/// Boot sweep: crash leftovers in the root + one-time legacy temp-dir cleanup.
///
/// Runs at real daemon boot AFTER the stores are open (the run-root sweep is
/// liveness-guarded against the session store). Never deletes the root itself.
#[cfg(unix)]
pub async fn boot_sweep() {
    if let Some(root) = temp_root() {
        // Crash leftovers: `.agent` spill/bg files from a previous daemon
        // session. THE startup purge — `agent_temp_dir` deliberately has no
        // once-flag purge anymore (overlapping mechanisms; the eager boot
        // sweep runs first and is the single mechanism).
        let agent_dir = root.join(".agent");
        if let Ok(mut entries) = tokio::fs::read_dir(&agent_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let _ = tokio::fs::remove_file(entry.path()).await;
            }
        }
        // Dead run-root folders (liveness-guarded — live runs are preserved).
        if let Ok(n) = crate::research_cleanup::sweep_run_roots().await
            && n > 0
        {
            tracing::info!(n, "Boot sweep: reclaimed dead research run folders");
        }
    }
    one_time_legacy_cleanup().await;
}

/// Clean the mahbot-managed bases left by previous generations in the legacy
/// (pre-pin) temp dir.
///
/// Two tiers:
/// - NON-liveness-gated bases (`.agent`, `mahbot_voice`, `mahbot_telegram_files`)
///   are removed exactly ONCE (marker-gated: the marker lives in the storage
///   root `~/.mahbot/.legacy-temp-cleaned`, which survives OS temp sweeps). The
///   marker is written ONLY when every removal succeeds — a failure (e.g. a
///   busy file) leaves no marker so the cleanup retries on the next boot
///   instead of being silently skipped forever.
/// - The legacy `mahbot-research` base is re-swept EVERY boot (cheap, usually
///   empty) with a liveness-guarded sweep: live pre-upgrade run folders are
///   preserved (prototypes are never destroyed by the migration), and folders
///   whose runs have since terminalized are reclaimed. NOT marker-gated — a
///   preserved-then-terminalized pre-upgrade folder must not orphan forever
///   just because the once-marker was already written.
async fn one_time_legacy_cleanup() {
    let Some(legacy) = legacy_temp_dir() else {
        return;
    };
    // Storage root for the once-only marker (CONFIG is loaded by the time the
    // boot sweep runs).
    let Some(storage_root) = crate::config::CONFIG
        .try_storage_root()
        .or_else(|| crate::config::default_config_dir().ok())
    else {
        return;
    };
    let marker = storage_root.join(".legacy-temp-cleaned");
    let already_cleaned = tokio::fs::try_exists(&marker).await.unwrap_or(false);

    if !already_cleaned {
        // `.agent` — daemon-owned spill/bg leftovers (recreated on demand).
        if !remove_if_exists(&legacy.join(".agent")).await {
            return;
        }
        // `mahbot_voice` / `mahbot_telegram_files` — daemon-owned ephemeral bases.
        if !remove_if_exists(&legacy.join("mahbot_voice")).await {
            return;
        }
        if !remove_if_exists(&legacy.join(crate::util::TELEGRAM_FILES_DIR)).await {
            return;
        }
        if let Err(e) = tokio::fs::write(&marker, b"legacy temp-dir bases cleaned once\n").await {
            tracing::warn!(error = %e, "Legacy temp cleanup: marker write failed — retrying next boot");
            return;
        }
        tracing::info!(legacy = %legacy.display(), "Legacy temp-dir bases cleaned once");
    }

    // EVERY boot: liveness-guarded sweep of the legacy research base. Live
    // pre-upgrade run folders (jobs rows still exist) are preserved; dead
    // leftovers are reclaimed — including folders preserved on earlier boots
    // whose runs have since terminalized.
    let legacy_research = legacy.join("mahbot-research");
    match crate::research_cleanup::sweep_run_roots_at(&legacy_research).await {
        Ok(n) if n > 0 => {
            tracing::info!(n, "Legacy temp cleanup: reclaimed dead legacy run folders");
        }
        Ok(_) => {}
        Err(e) => {
            tracing::warn!(error = %e, "Legacy temp cleanup: run-root sweep failed — retrying next boot");
        }
    }
}

/// Remove a legacy base directory; `NotFound` counts as cleaned. Returns
/// false (and logs) on any other failure so the once-only marker is withheld.
async fn remove_if_exists(dir: &std::path::Path) -> bool {
    match tokio::fs::remove_dir_all(dir).await {
        Ok(()) => true,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => true,
        Err(e) => {
            tracing::warn!(path = %dir.display(), error = %e, "Legacy temp cleanup: removal failed — retrying next boot");
            false
        }
    }
}

#[cfg(not(unix))]
pub async fn boot_sweep() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn legacy_capture_and_pin_are_consistent() {
        // init_temp_root is a process-global singleton — not run in tests
        // (it would pin TMPDIR for the whole test process). Just verify the
        // accessors are coherent: unset → no root, no legacy.
        assert!(temp_root().is_none());
        assert!(legacy_temp_dir().is_none());
        assert_eq!(shell_tmpdir(), "/tmp");
    }
}
