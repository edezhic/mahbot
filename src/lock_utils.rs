//! Shared lock utilities for single-instance enforcement.
//!
//! Provides a cross-platform `try_flock` that wraps `flock()` (Unix) or
//! `LockFileEx` (Windows) in a non-blocking exclusive lock attempt.
//!
//! Also provides `lock_file_path()` for the standard `mahbot.lock` path.
//!
//! # Design
//!
//! The core [`try_flock`] returns `io::Result<bool>` — the lowest common
//! denominator error type.  Callers in [`crate::self_update`] wrap the result
//! with `anyhow::Context` for richer error messages.
//!
//! # Lock release guarantee
//!
//! The returned `Ok(true)` from [`try_flock`] means the lock was acquired on
//! the given `File`.  The kernel releases the lock when the file descriptor is
//! closed during process teardown — this happens on normal `Drop`, on
//! `process::exit` (Rust destructors do NOT run, but the kernel closes all
//! fds), and on SIGKILL.  There is no stale-lock scenario.

use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

/// Path to the standard `mahbot.lock` under the given storage root.
///
/// Used by [`crate::self_update`] for the instance lock.
#[must_use]
pub fn lock_file_path(storage_root: &Path) -> PathBuf {
    storage_root.join("mahbot.lock")
}

/// Try to acquire an exclusive file lock non-blockingly.
///
/// Returns:
/// - `Ok(true)` — lock acquired.
/// - `Ok(false)` — lock held by another process.
/// - `Err(io::Error)` — non-retryable OS error.
///
/// # Panics
///
/// Panics on unsupported platforms (requires `cfg(unix)` or `cfg(windows)`).
#[cfg(unix)]
pub fn try_flock(file: &File) -> io::Result<bool> {
    use std::os::unix::io::AsRawFd;

    let fd = file.as_raw_fd();
    // SAFETY: `flock` operates on a valid fd opened with read+write.
    // `LOCK_NB` makes it non-blocking; `EAGAIN` means lock held.
    let result = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    if result == 0 {
        Ok(true)
    } else {
        let err = io::Error::last_os_error();
        match err.raw_os_error() {
            Some(libc::EAGAIN) => Ok(false),
            _ => Err(err),
        }
    }
}

/// Try to acquire an exclusive file lock non-blockingly (Windows).
#[cfg(windows)]
pub fn try_flock(file: &File) -> io::Result<bool> {
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
        let err = io::Error::last_os_error();
        match err.raw_os_error() {
            Some(LOCK_VIOLATION) => Ok(false),
            _ => Err(err),
        }
    }
}
