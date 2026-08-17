//! Private daemon temp root: one `/tmp/mahbot` root for ALL daemon temp
//! artifacts.
//!
//! ## One root
//!
//! The daemon creates a single private root under `/tmp` (`/tmp/mahbot`,
//! mode 0700) with exclusive-create + ownership/mode verification, and fails
//! loudly on a squatted path. It then pins its own `TMPDIR` to that root at
//! the very start of startup (before config and any temp use), so every
//! `std::env::temp_dir()`-based consumer relocates automatically: shell spill
//! files, research run folders, background-mode output, voice/telegram temp.
//! The OS removes stale files on its periodic temp sweep — the daemon builds
//! no startup/periodic reclamation of its own (crash leftovers are the
//! operating system's job), so the root is simply recreated/re-verified on
//! every boot.
//!
//! The path is fixed (no per-user `-{uid}` suffix): this is a single-user
//! deployment, and the suffix was superfluous. The accepted multi-user
//! consequence: with a shared fixed path, a second OS user's daemon fails
//! loudly at boot via the ownership check below — accepted for this
//! deployment. Old suffixed leftovers (`/tmp/mahbot-<uid>`) are left to the
//! OS sweep.
//!
//! Shell children get `TMPDIR` set to the same root (see
//! [`crate::tools::shell::shell_tmpdir`]).
//!
//! Run folders inside the root (`mahbot-research/{job_id}`) are removed by
//! the run's own completion flow (see `crate::research_cleanup`); nothing is
//! reclaimed at daemon startup.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The pinned private temp root, set once by [`init_temp_root`].
static TEMP_ROOT: OnceLock<PathBuf> = OnceLock::new();

/// The pre-pin OS temp dir (e.g. `/var/folders/.../T` on macOS). Captured
/// before the `TMPDIR` pin so the readonly guard keeps it as an allowed root
/// (bare macOS `mktemp` ignores `TMPDIR` and lands there).
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
/// create the root). Unix-only: `/tmp/mahbot` is meaningless on Windows,
/// where the shell env uses `TEMP`/`TMP` instead.
///
/// Failure modes (fail loudly, never paper over):
/// - the root exists but is not a directory;
/// - the root exists and is owned by a different uid (squatting — with the
///   fixed shared path this is also the second-OS-user guard: their daemon
///   fails loudly here, the accepted multi-user consequence);
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
    // Fixed shared path (no `-{uid}` suffix): single-user deployment; the
    // ownership check below is what makes the shared path safe — a second OS
    // user's daemon fails loudly at boot instead of sharing the root.
    let root = PathBuf::from("/tmp/mahbot");

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
