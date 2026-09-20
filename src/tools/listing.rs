//! The compact directory-listing format agents see: directories first with a
//! trailing `/`, then files with their size, then an extension summary. The
//! `read` tool renders it from its own walk of the filesystem and the shell's
//! `ls` output profile from `ls -l` text, so both routes share one format.

use std::collections::HashMap;
use std::fmt::Write;

/// One entry of a directory listing.
pub(crate) struct ListingEntry {
    /// Entry name, without the directory mark and without any link target.
    pub name: String,
    /// Directories are grouped first and rendered with a trailing `/`.
    pub is_dir: bool,
    /// Display size of a file entry, or `None` when the platform could not
    /// measure it — rendered as `?`, the size `ls` itself prints for a file it
    /// cannot stat. Directories carry none: their size is not rendered.
    pub size: Option<String>,
}

/// Format a byte count into the listing's size string (decimal units).
#[expect(clippy::cast_precision_loss)]
pub(crate) fn human_readable_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}G", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.1}M", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1}K", bytes as f64 / 1_000.0)
    } else {
        format!("{bytes}B")
    }
}

/// Render the compact listing: directories first (each with a trailing `/`),
/// then files with their size, then a summary line carrying the file/directory
/// counts and the entry counts of the five commonest extensions.
///
/// The order the caller supplies is preserved within each group. An empty
/// listing renders as `(empty)`.
pub(crate) fn format_listing(entries: &[ListingEntry]) -> String {
    let mut dirs: Vec<&str> = Vec::new();
    let mut files: Vec<(&str, &str)> = Vec::new();
    let mut ext_counts: HashMap<String, usize> = HashMap::new();

    for entry in entries {
        if entry.is_dir {
            dirs.push(&entry.name);
            continue;
        }
        let ext = entry
            .name
            .rsplit_once('.')
            .map_or_else(|| "no ext".to_string(), |(_, ext)| format!(".{ext}"));
        *ext_counts.entry(ext).or_insert(0) += 1;
        files.push((&entry.name, entry.size.as_deref().unwrap_or("?")));
    }

    if dirs.is_empty() && files.is_empty() {
        return "(empty)\n".to_string();
    }

    let mut listing = String::new();

    for dir in &dirs {
        let _ = writeln!(listing, "{dir}/");
    }

    for (name, size) in &files {
        let _ = writeln!(listing, "{name}  {size}");
    }

    let _ = write!(
        listing,
        "Summary: {} files, {} dirs",
        files.len(),
        dirs.len()
    );
    if !ext_counts.is_empty() {
        let mut sorted: Vec<_> = ext_counts.iter().collect();
        // Equally common extensions break by name: `HashMap` iteration order is
        // arbitrary, which would make both the order and *which* extensions reach
        // the five shown unstable, and the listing must be byte-stable.
        sorted.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        let parts: Vec<String> = sorted
            .iter()
            .take(5)
            .map(|(ext, count)| format!("{count} {ext}"))
            .collect();
        let _ = write!(listing, " ({})", parts.join(", "));
        if sorted.len() > 5 {
            let _ = write!(listing, ", +{} more", sorted.len() - 5);
        }
    }
    listing.push('\n');

    listing
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(name: &str, size: u64) -> ListingEntry {
        ListingEntry {
            name: name.to_string(),
            is_dir: false,
            size: Some(human_readable_size(size)),
        }
    }

    #[test]
    fn directories_group_first_with_a_trailing_mark() {
        let listing = format_listing(&[
            file("main.rs", 2048),
            ListingEntry {
                name: "src".to_string(),
                is_dir: true,
                size: None,
            },
            file(".gitignore", 512),
        ]);
        assert_eq!(
            listing,
            "src/\nmain.rs  2.0K\n.gitignore  512B\nSummary: 2 files, 1 dirs (1 .gitignore, 1 .rs)\n"
        );
    }

    #[test]
    fn sizes_use_decimal_units() {
        assert_eq!(human_readable_size(0), "0B");
        assert_eq!(human_readable_size(999), "999B");
        assert_eq!(human_readable_size(1_000), "1.0K");
        assert_eq!(human_readable_size(1_500_000), "1.5M");
        assert_eq!(human_readable_size(2_000_000_000), "2.0G");
    }

    /// Extensions beyond the fifth are collapsed into a remainder count.
    #[test]
    fn summary_keeps_five_extensions_and_counts_the_rest() {
        let listing = format_listing(&[
            file("a.rs", 1),
            file("b.rs", 1),
            file("c.py", 1),
            file("d.md", 1),
            file("e.toml", 1),
            file("f.txt", 1),
            file("g.json", 1),
        ]);
        assert_eq!(
            listing.lines().last().unwrap(),
            "Summary: 7 files, 0 dirs (2 .rs, 1 .json, 1 .md, 1 .py, 1 .toml), +1 more"
        );
    }
}
