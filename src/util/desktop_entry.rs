//! An `Exec` field value, shared by the two desktop entries the product writes:
//! the launcher entry and the autostart entry.

use std::path::Path;

/// A path as an `Exec` argument: quoted, with the characters the desktop entry
/// specification reserves inside a quoted argument escaped and `%` doubled — it
/// introduces a field code there, and a literal one is written `%%`. The file can
/// sit anywhere, spaces and percentages included.
pub(crate) fn exec_argument(path: &Path) -> String {
    let path = path.to_string_lossy();
    let mut argument = String::with_capacity(path.len() + 2);
    argument.push('"');
    for character in path.chars() {
        match character {
            '"' | '\\' | '$' | '`' => {
                argument.push('\\');
                argument.push(character);
            }
            '%' => argument.push_str("%%"),
            _ => argument.push(character),
        }
    }
    argument.push('"');
    argument
}
