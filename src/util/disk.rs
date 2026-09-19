//! Disk free-space readings.
//!
//! One reading, spelled per platform: `statvfs` on unix, `GetDiskFreeSpaceExW`
//! on Windows, `None` elsewhere. Both report the space available to the calling
//! user, not the volume's raw free blocks — a per-user quota is what actually
//! bounds writes into the user's temp directory, so a reading that ignored it
//! would let callers walk into the quota wall while the disk still looked roomy.

use std::path::Path;

/// `(free, capacity)` bytes on the volume holding `path`; `None` when the
/// reading is unavailable (a failed `statvfs`, or an unusable path). `free` is
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

/// Whether a wide path needs a trailing separator before it reaches the
/// free-space query: true for a UNC-shaped path that lacks one. The API reads a
/// UNC directory only when it is spelled with a trailing separator, and a UNC
/// storage root is reachable on a roaming-profile machine. The verbatim
/// (`\\?\C:\…`) and device (`\\.\…`) forms share the leading double backslash
/// but are not UNC names, so the third character decides.
#[cfg(any(windows, test))]
#[must_use]
fn needs_unc_separator(wide: &[u16]) -> bool {
    let backslash = u16::from(b'\\');
    wide.starts_with(&[backslash, backslash])
        && !matches!(wide.get(2), Some(&c) if c == u16::from(b'?') || c == u16::from(b'.'))
        && !wide.ends_with(&[backslash])
        && !wide.ends_with(&[u16::from(b'/')])
}

/// `(free, capacity)` bytes on the volume holding `path`; `None` when the
/// reading is unavailable (a failed `GetDiskFreeSpaceExW`, or a path with an
/// interior NUL, which would silently name a different volume). `free` is
/// `lpFreeBytesAvailableToCaller` — the space available to the calling user, so
/// a per-user quota is honoured — and `capacity` is `lpTotalNumberOfBytes`.
#[cfg(windows)]
#[must_use]
pub(crate) fn free_and_capacity(path: &Path) -> Option<(u64, u64)> {
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

    let mut wide = crate::util::wide_path(path)?;
    if needs_unc_separator(&wide) {
        wide.push(u16::from(b'\\'));
    }
    wide.push(0);
    let mut free = 0_u64;
    let mut capacity = 0_u64;
    // SAFETY: `wide` is NUL-terminated and outlives the call; each out-parameter
    // is a valid `u64` for the documented write, and the third is explicitly
    // null, which the API accepts.
    let ok = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            std::ptr::addr_of_mut!(free),
            std::ptr::addr_of_mut!(capacity),
            std::ptr::null_mut(),
        )
    };
    (ok != 0).then_some((free, capacity))
}

/// No free-space query on a platform that is neither unix nor Windows.
#[cfg(not(any(unix, windows)))]
#[must_use]
pub(crate) fn free_and_capacity(_path: &Path) -> Option<(u64, u64)> {
    None
}

#[cfg(test)]
mod tests {
    use super::needs_unc_separator;

    /// A wide path as `OsStr::encode_wide` would produce it.
    fn wide(path: &str) -> Vec<u16> {
        path.encode_utf16().collect()
    }

    #[test]
    fn only_a_unc_shaped_path_needs_the_trailing_separator() {
        // A UNC root is read only with a trailing separator.
        assert!(needs_unc_separator(&wide(r"\\srv\share")));
        // Already spelled with one — appending a second would break the read.
        assert!(!needs_unc_separator(&wide(r"\\srv\share\")));
        assert!(!needs_unc_separator(&wide(r"\\srv/share/")));
        // The verbatim and device forms share the double backslash but are not
        // UNC names.
        assert!(!needs_unc_separator(&wide(r"\\?\C:\ws")));
        assert!(!needs_unc_separator(&wide(r"\\.\C:")));
        // An ordinary local path is passed through untouched.
        assert!(!needs_unc_separator(&wide(r"C:\ws")));
    }
}
