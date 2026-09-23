//! The one rule for a name the product turns into a folder.
//!
//! A folder name must be creatable on every platform the product runs on, and
//! the storage it builds is portable between them: a name that works on one
//! platform and fails on another is exactly the defect this rule exists to
//! prevent. So the rule is the strict intersection of every platform's limits,
//! applied identically everywhere rather than switched per platform.
//!
//! Only the impossible is refused; an accepted name is used verbatim, because
//! callers read the folder name back as the name it was built from.
//!
//! The rule bounds one path component, the thing a name becomes; it is not a
//! bound on the whole path, because the standard library re-issues an over-long
//! absolute path in verbatim form for every filesystem call (and `tokio::fs`
//! delegates to it), which lifts Windows' classic 260-character path ceiling.

/// The rule in the words the surfaces that propose a name use, so that what
/// they ask for is what [`folder_name_problem`] accepts.
pub(crate) const RULE_SUMMARY: &str = "not empty, not only dots, no / \\ : * ? \" < > |, no \
     control characters, nothing ending in a dot or a space, no reserved device name (CON, NUL, \
     PRN, AUX, COM1-9, LPT1-9 — also with an extension or a superscript digit, plus CONIN$ and \
     CONOUT$), and at most 255 bytes";

/// Longest accepted name, in bytes of UTF-8 — the tightest per-component limit
/// among the platforms the product runs on. A byte count is the safe side of
/// the comparison: for any name, its UTF-8 byte count is never below its UTF-16
/// code-unit count, so a name within this many bytes is within the 255-unit
/// component limit of the other platforms as well.
const MAX_NAME_BYTES: usize = 255;

/// Windows' reserved device names, which address a device instead of a file in
/// every directory: the documented set, the superscript COM¹/COM²/COM³ and
/// LPT¹/²/³ spellings its digit test also matches, and the console names.
const RESERVED_DEVICE_NAMES: [&str; 30] = [
    "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8",
    "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9", "com¹", "com²",
    "com³", "lpt¹", "lpt²", "lpt³", "conin$", "conout$",
];

/// Why `name` cannot be a directory name on every platform the product runs
/// on, in the words the refusal uses; `None` when it can.
#[must_use]
pub(crate) fn folder_name_problem(name: &str) -> Option<&'static str> {
    if name.is_empty() {
        return Some("it is empty");
    }
    // Every dot-only name also ends in a dot — this branch is here for its
    // message, which names the real problem instead of the last character.
    if name.chars().all(|c| c == '.') {
        return Some("it consists only of dots");
    }
    if name.len() > MAX_NAME_BYTES {
        return Some("it is longer than 255 bytes");
    }
    if name.contains(['/', '\\']) {
        return Some("it contains a path separator");
    }
    if name.contains(|c: char| c <= '\u{1f}') {
        return Some("it contains a control character");
    }
    if name.contains(['<', '>', ':', '"', '|', '?', '*']) {
        return Some("it contains a character no folder name may hold");
    }
    if name.ends_with(['.', ' ']) {
        return Some("it ends with a dot or a space");
    }
    if is_reserved_device_name(name) {
        return Some("it is a name the platform reserves for a device");
    }
    None
}

/// Whether `name` names a reserved Windows device. Matching is
/// case-insensitive and stops at the name's first dot: the platform treats a
/// device name followed by an extension as the device itself (`NUL.txt` is
/// `NUL`).
fn is_reserved_device_name(name: &str) -> bool {
    let stem = name.split_once('.').map_or(name, |(stem, _)| stem);
    RESERVED_DEVICE_NAMES.contains(&stem.to_lowercase().as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule accepts exactly the names every platform can hold: each refused
    /// name fails on at least one of them, each accepted name is creatable
    /// everywhere — and accepted names are never rewritten, so a name the rule
    /// lets through reaches the filesystem byte-identically.
    #[test]
    fn folder_name_rule_is_the_cross_platform_intersection() {
        let too_long = "x".repeat(MAX_NAME_BYTES + 1);
        let refused: [&str; 34] = [
            "",
            ".",
            "..",
            "...",
            "a/b",
            "a\\b",
            "/etc",
            "../evil",
            "..\\evil",
            "C:\\x",
            "C:x",
            "a\u{1}b",
            "a\u{0}b",
            "a:b",
            "a?b",
            "a*b",
            "a\"b",
            "a<b",
            "a>b",
            "a|b",
            "trailing.",
            "trailing ",
            "   ",
            "nul",
            "NUL",
            "nul.txt",
            "con.tar.gz",
            "Com1",
            "lpt9",
            "com¹",
            "LPT³",
            "conin$",
            "CONOUT$",
            &too_long,
        ];
        for name in refused {
            assert!(
                folder_name_problem(name).is_some(),
                "must be refused: {name:?}"
            );
        }

        let longest = "x".repeat(MAX_NAME_BYTES);
        let accepted: [&str; 12] = [
            "admin",
            "alice",
            "a b",
            "café",
            "a.b",
            "conway",
            "null",
            ".hidden",
            "x",
            "a-b_c",
            "Japanese漢字",
            &longest,
        ];
        for name in accepted {
            assert_eq!(
                folder_name_problem(name),
                None,
                "must be accepted: {name:?}"
            );
        }
    }
}
