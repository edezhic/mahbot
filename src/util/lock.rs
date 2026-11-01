//! Shared lock utilities for single-instance enforcement.
//!
//! [`try_flock`] is a non-blocking exclusive lock on the WHOLE file, over std's [`File::try_lock`]:
//! `flock(LOCK_EX|LOCK_NB)` on Unix, `LockFileEx` over the whole addressable range from offset 0
//! on Windows. Std owning both the byte range and the platform's contention mapping is deliberate
//! — a whole-file Windows range cannot overlap whatever another handle holds.
//!
//! [`lock_file_path`] gives the standard `mahbot.lock` path. The probes at the bottom answer "does
//! an instance hold this storage location?" for the read-only CLI paths: they report a tri-state,
//! because "the probe could not tell" must never be read as "no instance holds the location".
//!
//! # Release
//!
//! The lock is released by an explicit [`File::unlock`], or at the latest when the last handle to
//! the file is closed: on `Drop`, on `process::exit` (destructors do not run, but the O.S. closes
//! all handles), and on abrupt termination. There is no stale-lock scenario. The explicit unlock
//! is what [`crate::self_update`] uses before spawning the replacement instance: Windows documents
//! the O.S. unlock on close as taking an unspecified amount of time.

use std::fs::{File, TryLockError};
use std::io;
use std::path::{Path, PathBuf};

/// Path to the standard `mahbot.lock` under the given storage root.
#[must_use]
pub(crate) fn lock_file_path(storage_root: &Path) -> PathBuf {
    storage_root.join("mahbot.lock")
}

/// Try to acquire an exclusive whole-file lock non-blockingly: `Ok(true)` acquired, `Ok(false)`
/// already held, `Err` a non-retryable OS error.
pub(crate) fn try_flock(file: &File) -> io::Result<bool> {
    match file.try_lock() {
        Ok(()) => Ok(true),
        Err(TryLockError::WouldBlock) => Ok(false),
        Err(TryLockError::Error(e)) => Err(e),
    }
}

/// What the instance lock says about a storage location.
pub(crate) enum InstanceLockState {
    /// No instance holds the location: a direct read-only open is safe.
    Free,
    /// Another process holds the location (an instance is running).
    Held,
    /// The probe could not tell. Callers must never treat this as [`InstanceLockState::Free`].
    Unknown(io::Error),
}

/// Refusal wording for the read-only CLI paths (`mahbot debug`, `bench-openrouter`) when the probe
/// reported [`Unknown`](InstanceLockState::Unknown): whether an instance holds the storage location
/// could not be established, so the probe refuses. Returned as a clause without terminal
/// punctuation — each consumer appends its own continuation.
pub(crate) fn lock_state_unknown(root: &Path, e: &io::Error) -> String {
    format!(
        "cannot tell whether a mahbot instance holds {} ({e}); the live store was left untouched \
         instead of being opened directly",
        root.display()
    )
}

/// Probe the instance lock once for the given storage location: open it read+write and
/// [`try_flock`] it once.
///
/// A `NotFound` open error is [`InstanceLockState::Free`] (the lock file is created before the
/// stores are opened), any other open error [`Unknown`]; `Ok(true)` is `Free` — the probe releases
/// what it took — `Ok(false)` is `Held`, and `Err` is `Unknown`.
///
/// A held lock means an instance is running: the caller must NOT open the live stores directly and
/// must route read-only queries through the debug IPC endpoint. [`InstanceLockState::Unknown`] is
/// deliberately never read as "free".
fn instance_lock_state(storage_root: &Path) -> InstanceLockState {
    let lock_path = lock_file_path(storage_root);
    let file = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&lock_path)
    {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return InstanceLockState::Free,
        Err(e) => return InstanceLockState::Unknown(e),
    };
    match try_flock(&file) {
        Ok(true) => {
            // The probe took the lock: release it explicitly rather than leave it to the close. A
            // failed unlock only defers the release, so it is not worth reporting.
            let _ = file.unlock();
            InstanceLockState::Free
        }
        Ok(false) => InstanceLockState::Held,
        Err(e) => InstanceLockState::Unknown(e),
    }
}

/// Re-check count/interval for [`instance_lock_state_settled`].
const LOCK_SETTLE_RECHECKS: usize = 3;
const LOCK_SETTLE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);

/// Conservative instance-up probe for read-only CLI fallbacks (`mahbot debug`, `bench-openrouter`):
/// re-probes the instance lock over a short window.
///
/// [`Held`](InstanceLockState::Held) wins outright and is returned as soon as it is seen, which
/// keeps the caller from direct-opening a live store during the self-update handoff.
///
/// [`Free`](InstanceLockState::Free) is returned only after the window consistently shows the lock
/// free. An [`Unknown`](InstanceLockState::Unknown) seen in the window is remembered and returned
/// if no `Held` was ever seen: a probe that could not tell must never be downgraded to `Free`. The
/// extra ~300 ms is a CLI diagnostic tool, not a hot path.
#[must_use]
pub(crate) fn instance_lock_state_settled(storage_root: &Path) -> InstanceLockState {
    let mut unknown = None;
    for i in 0..=LOCK_SETTLE_RECHECKS {
        match instance_lock_state(storage_root) {
            InstanceLockState::Held => return InstanceLockState::Held,
            InstanceLockState::Free => {}
            InstanceLockState::Unknown(e) => unknown = Some(e),
        }
        if i < LOCK_SETTLE_RECHECKS {
            std::thread::sleep(LOCK_SETTLE_INTERVAL);
        }
    }
    unknown.map_or(InstanceLockState::Free, InstanceLockState::Unknown)
}
