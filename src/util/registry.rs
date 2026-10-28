//! The registry's value shapes, shared by the modules that read and write them:
//! the UTF-16 the API takes and gives back, and the sentence a failed call
//! reports.

/// `text` as the NUL-terminated UTF-16 the registry API takes.
pub(crate) fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}

/// `text` as the NUL-terminated UTF-16 little-endian bytes a `REG_SZ` holds.
pub(crate) fn utf16_bytes(text: &str) -> Vec<u8> {
    wide(text).into_iter().flat_map(u16::to_le_bytes).collect()
}

/// The text of a registry string value: UTF-16, little-endian, with the
/// terminating NUL the API stores dropped.
pub(crate) fn utf16_text(data: &[u8]) -> String {
    let units: Vec<u16> = data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect();
    String::from_utf16_lossy(&units)
        .trim_end_matches('\0')
        .to_string()
}

/// A failure's reason: `what` with only the OS error code appended.
#[must_use]
pub(crate) fn os_reason(what: &str, code: u32) -> String {
    format!("{what} (os error {code})")
}
