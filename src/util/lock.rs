//! Shared lock utilities for single-instance enforcement.
//!
//! [`try_flock`] is a non-blocking exclusive lock on the WHOLE file, over std's
//! [`File::try_lock`]: `flock(LOCK_EX|LOCK_NB)` on Unix, `LockFileEx` over
//! `u32::MAX` bytes low and high (the whole addressable range) from offset 0 on
//! Windows. Std owning both the byte range and the platform's contention mapping
//! is deliberate — a whole-file Windows range cannot overlap whatever another
//! handle holds, and std knows which code that comes back as.
//!
//! [`lock_file_path`] gives the standard `mahbot.lock` path. The probes at the
//! bottom answer "does an instance hold this storage location?" for the
//! read-only CLI paths: they report a tri-state, because "the probe could not
//! tell" must never be read as "no instance holds the location".
//!
//! # Release
//!
//! The lock is released by an explicit [`File::unlock`], or at the latest when
//! the last handle to the file is closed: on `Drop`, on `process::exit` (Rust
//! destructors do not run, but the O.S. closes all handles), and on abrupt
//! termination. There is no stale-lock scenario. The explicit unlock is what
//! [`crate::self_update`] uses before spawning the replacement instance: it must
//! be able to take the lock as soon as the release returns, and neither platform
//! guarantees that a close-only release has landed by then — Windows documents
//! the O.S. unlock on close as taking an unspecified amount of time, depending
//! on available system resources.

use std::fs::{File, TryLockError};
use std::io;
use std::path::{Path, PathBuf};

/// Path to the standard `mahbot.lock` under the given storage root.
///
/// Used by [`crate::self_update`] for the instance lock.
#[must_use]
pub(crate) fn lock_file_path(storage_root: &Path) -> PathBuf {
    storage_root.join("mahbot.lock")
}

/// Try to acquire an exclusive whole-file lock non-blockingly.
///
/// Returns:
/// - `Ok(true)` — lock acquired.
/// - `Ok(false)` — lock held by another process.
/// - `Err(io::Error)` — non-retryable OS error.
pub fn try_flock(file: &File) -> io::Result<bool> {
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
    /// The probe could not tell (the lock file could not be opened or locked).
    /// Callers must never treat this as [`InstanceLockState::Free`].
    Unknown(io::Error),
}

/// Refusal wording for the read-only CLI paths (`mahbot debug`,
/// `bench-openrouter`) when the probe reported
/// [`Unknown`](InstanceLockState::Unknown): whether an instance holds the
/// storage location could not be established, so the probe — the thing that
/// decides whether opening a store is safe — refuses, and both halves show in
/// the sentence. Shared by both consumers so they cannot drift apart, and
/// returned as a clause without terminal punctuation: each appends its own
/// continuation (`mahbot debug` its re-run advice, the bench what it uses
/// instead).
pub(crate) fn lock_state_unknown(root: &Path, e: &io::Error) -> String {
    format!(
        "cannot tell whether a mahbot instance holds {} ({e}); the live store was left untouched \
         instead of being opened directly",
        root.display()
    )
}

/// Probe the instance lock once for the given storage location.
///
/// The lock file is opened read+write and [`try_flock`]ed once:
///
/// * A `NotFound` open error is [`InstanceLockState::Free`] — no instance ever
///   took this location (the lock file is created before the stores are opened).
/// * Any other open error is [`InstanceLockState::Unknown`].
/// * `Ok(true)` from [`try_flock`] is [`InstanceLockState::Free`] — the probe
///   acquires and immediately releases the lock, so it never leaves it held.
/// * `Ok(false)` is [`InstanceLockState::Held`].
/// * `Err(e)` from [`try_flock`] is [`InstanceLockState::Unknown`].
///
/// A held lock means an instance is running: the caller must NOT open the live
/// stores directly (single-process mode would conflict with that instance's
/// writer) and must route read-only queries through the debug IPC endpoint
/// instead. A free lock means no instance holds the location, so a direct
/// single-process read-only open is safe. [`InstanceLockState::Unknown`] is
/// deliberately never read as "free" — a lock file the inspecting user cannot
/// open read+write makes the read-only CLI refuse where it would otherwise have
/// read the store directly.
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
            // The probe took the lock: release it explicitly rather than leave
            // the release to the close (see the module doc's Release section). A
            // failed unlock only defers the release to that close, so it is not
            // worth reporting.
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

/// Conservative instance-up probe for read-only CLI fallbacks (`mahbot debug`,
/// `bench-openrouter`).
///
/// Re-probes the instance lock over a short window.
/// [`Held`](InstanceLockState::Held) wins outright and is returned as soon as it
/// is seen — a held lock means an instance is running, so there is nothing left
/// to settle. This prevents the caller from direct-opening a live store during
/// the self-update handoff, when the outgoing instance has released the lock but
/// the incoming one has not yet re-acquired it (a sub-millisecond window).
///
/// [`Free`](InstanceLockState::Free) is returned only after the window
/// consistently shows the lock free — no instance holds the location. An
/// [`Unknown`](InstanceLockState::Unknown) observed in the window is remembered
/// and returned if no `Held` was ever seen: the probe may have failed on an
/// ordinary racing window, and a probe that could not tell must never be
/// downgraded to `Free`. The extra ~300 ms latency is a CLI diagnostic tool, not
/// a hot path.
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
