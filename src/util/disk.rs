//! Disk free-space readings.

use std::path::Path;

/// `(free, capacity)` bytes on the volume holding `path`; `None` when the
/// reading is unavailable (non-unix, or a failed `statvfs`). `free` is
/// `f_bavail` — the space available to an unprivileged process, not `f_bfree`.
#[cfg(unix)]
#[must_use]
pub(crate) fn free_and_capacity(path: &Path) -> Option<(u64, u64)> {
    use std::os::unix::ffi::OsStrExt;
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stats = unsafe { std::mem::zeroed::<libc::statvfs>() };
    if unsafe { libc::statvfs(c_path.as_ptr(), std::ptr::addr_of_mut!(stats)) } != 0 {
        return None;
    }
    let frsize = stats.f_frsize;
    if frsize == 0 {
        return None;
    }
    Some((
        u64::from(stats.f_bavail).saturating_mul(frsize),
        u64::from(stats.f_blocks).saturating_mul(frsize),
    ))
}

/// Windows has no free-space query through libc.
#[cfg(not(unix))]
#[must_use]
pub(crate) fn free_and_capacity(_path: &Path) -> Option<(u64, u64)> {
    None
}
