//! Git diff parsing — parse unified diff output into structured data.
//!
//! Handles `git diff HEAD --no-color --find-renames` or `git show -m` output plus
//! untracked files from `git status --porcelain`.
//!
//! For git subprocess wrappers that produce diff output or manage
//! repository state, see [`crate::git_commands`].

use tracing::warn;

use crate::util::unquote_c_style;

/// Status of a file in the diff — what kind of change it represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffFileStatus {
    /// File exists on both sides (modified or unchanged).
    Modified,
    /// New file added to git (from `--- /dev/null`).
    Added,
    /// File deleted from git (from `+++ /dev/null`).
    Deleted,
    /// File renamed (from `rename from`/`rename to`).
    Renamed,
    /// Untracked file (from `git status --porcelain`).
    Untracked,
}

/// A single changed file in the diff.
#[derive(Debug, Clone)]
pub struct DiffFile {
    pub path: String,
    /// Previous path when git detects a rename. None otherwise.
    pub old_path: Option<String>,
    pub hunks: Vec<DiffHunk>,
    /// Change status (Added, Deleted, Renamed, Untracked, or Modified).
    pub status: DiffFileStatus,
    /// Whether the file is binary (orthogonal — not a status category).
    pub is_binary: bool,
    /// If too large to diff, the file size in bytes. None otherwise.
    pub too_large_size: Option<u64>,
}

impl DiffFile {
    /// Create a placeholder entry for binary or too-large untracked files.
    #[must_use]
    pub const fn placeholder(path: String, is_binary: bool, too_large_size: Option<u64>) -> Self {
        Self {
            path,
            old_path: None,
            hunks: Vec::new(),
            status: DiffFileStatus::Untracked,
            is_binary,
            too_large_size,
        }
    }
}

/// One hunk within a file diff.
#[derive(Debug, Clone)]
pub struct DiffHunk {
    /// The hunk header line, e.g. "@@ -10,7 +10,9 @@ fn main() {"
    pub header: String,
    pub lines: Vec<DiffLine>,
}

/// One line within a diff hunk or untracked file.
#[derive(Debug, Clone)]
pub struct DiffLine {
    pub kind: DiffLineKind,
    pub old_line_number: Option<usize>,
    pub new_line_number: Option<usize>,
    /// The code content (without the leading +/-/space prefix).
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffLineKind {
    Added,
    Removed,
    Context,
}

impl DiffLineKind {
    /// The leading character used in unified diff format: '+', '-', or ' '.
    #[must_use]
    pub const fn prefix(self) -> char {
        match self {
            Self::Added => '+',
            Self::Removed => '-',
            Self::Context => ' ',
        }
    }
}

#[derive(Default)]
struct DiffParser {
    files: Vec<DiffFile>,
    current_file: Option<DiffFile>,
    current_hunk: Option<DiffHunk>,
    old_counter: usize,
    new_counter: usize,
}

impl DiffParser {
    /// Flush any pending hunk into the current file (if both exist).
    fn flush_hunk(&mut self) {
        if let Some(hunk) = self.current_hunk.take()
            && let Some(f) = &mut self.current_file
        {
            f.hunks.push(hunk);
        }
    }

    /// Flush the current file (with optional pending hunk) into the files vec.
    fn flush(&mut self) {
        self.flush_hunk();
        if let Some(file) = self.current_file.take() {
            self.files.push(file);
        }
    }

    /// Handle a `diff --git` line: flush previous file, reset counters, and create a new file entry.
    fn handle_diff_git_header(&mut self, line: &str) {
        self.flush();
        self.old_counter = 0;
        self.new_counter = 0;

        if let Some(path) = parse_diff_git_line(line) {
            self.current_file = Some(DiffFile {
                path,
                old_path: None,
                hunks: Vec::new(),
                status: DiffFileStatus::Modified,
                is_binary: false,
                too_large_size: None,
            });
        }
    }

    /// Handle a `rename from` line: set rename status and parse the old path.
    /// Falls back to `Modified` status when the rename path is malformed.
    fn handle_rename_from(&mut self, line: &str) {
        let Some(f) = self.current_file.as_mut() else {
            return;
        };

        f.status = DiffFileStatus::Renamed;
        let Some(raw) = line.strip_prefix("rename from ") else {
            warn!(
                line = %line,
                "rename from: unexpected format, dropping rename info"
            );
            f.status = DiffFileStatus::Modified;
            return;
        };
        let Some(old_path) = unquote_c_style(raw) else {
            warn!(
                line = %line,
                "rename from: malformed C-style escape, dropping rename info"
            );
            f.status = DiffFileStatus::Modified;
            return;
        };
        f.old_path = Some(old_path);
    }

    /// Handle a hunk header line (`@@ -old,count +new,count @@`): flush previous hunk,
    /// parse the header, and set up a new hunk with fresh line counters.
    fn handle_hunk_header(&mut self, line: &str) {
        self.flush_hunk();
        let (old_start, new_start) = parse_hunk_header(line);
        self.old_counter = old_start;
        self.new_counter = new_start;
        self.current_hunk = Some(DiffHunk {
            header: line.to_string(),
            lines: Vec::new(),
        });
    }

    /// Handle a content line within a hunk: classify as Added/Removed/Context, track
    /// line numbers, and push to the hunk.
    ///
    /// Annotation lines (`\ No newline at end of file`) are filtered early, before the
    /// hunk guard, since they are not diff content. Unknown lines are also silently
    /// skipped — the caller continues normally.
    fn handle_diff_content_line(&mut self, line: &str) {
        if line == r"\ No newline at end of file" {
            return;
        }

        // If no hunk is active, this line falls outside any diff hunk — skip.
        let Some(hunk) = self.current_hunk.as_mut() else {
            return;
        };

        let line_kind = if line.starts_with('+') {
            DiffLineKind::Added
        } else if line.starts_with('-') {
            DiffLineKind::Removed
        } else if line.starts_with(' ') {
            DiffLineKind::Context
        } else {
            return;
        };

        let content = line[1..].trim_end_matches('\r');
        let (old_num, new_num) = match line_kind {
            DiffLineKind::Added => {
                let n = Some(self.new_counter);
                self.new_counter += 1;
                (None, n)
            }
            DiffLineKind::Removed => {
                let n = Some(self.old_counter);
                self.old_counter += 1;
                (n, None)
            }
            DiffLineKind::Context => {
                let o = Some(self.old_counter);
                let n = Some(self.new_counter);
                self.old_counter += 1;
                self.new_counter += 1;
                (o, n)
            }
        };

        hunk.lines.push(DiffLine {
            kind: line_kind,
            old_line_number: old_num,
            new_line_number: new_num,
            content: content.to_string(),
        });
    }

    /// Process a single line from the diff output.
    fn process_line(&mut self, line: &str) {
        if line.starts_with("diff --git ") {
            self.handle_diff_git_header(line);
        } else if line.starts_with("index ")
            || line.starts_with("new file mode ")
            || line.starts_with("deleted file mode ")
            || line.starts_with("old mode ")
            || line.starts_with("new mode ")
            || line.starts_with("rename to ")
        {
            // rename to is safe to skip because diff --git already captures the b-path,
            // and rename from already set the status.
        } else if line.starts_with("--- ") || line.starts_with("+++ ") {
            if let Some(ref mut f) = self.current_file {
                if line.starts_with("--- /dev/null") {
                    f.status = DiffFileStatus::Added;
                } else if line.starts_with("+++ /dev/null") {
                    f.status = DiffFileStatus::Deleted;
                }
            }
        } else if line.starts_with("rename from ") {
            self.handle_rename_from(line);
        } else if line.starts_with("Binary files ") {
            if let Some(ref mut f) = self.current_file {
                f.is_binary = true;
            }
        } else if line.starts_with("@@") {
            self.handle_hunk_header(line);
        } else {
            self.handle_diff_content_line(line);
        }
    }
}

/// Parse unified diff output (from `git diff HEAD` or `git show -m`).
#[must_use]
pub fn parse_git_diff(diff_output: &str) -> Vec<DiffFile> {
    let mut parser = DiffParser::default();
    for line in diff_output.lines() {
        parser.process_line(line);
    }
    parser.flush();
    parser.files
}

/// Create a [`DiffFile`] for an untracked file (showing all lines as added).
#[must_use]
pub fn make_untracked_diff_file(path: &str, content: &str) -> DiffFile {
    let lines: Vec<DiffLine> = content
        .lines()
        .enumerate()
        .map(|(idx, line)| DiffLine {
            kind: DiffLineKind::Added,
            old_line_number: None,
            new_line_number: Some(idx + 1),
            content: line.trim_end_matches('\r').to_string(),
        })
        .collect();

    let hunk = DiffHunk {
        header: format!("@@ -0,0 +1,{} @@ new file", lines.len()),
        lines,
    };

    DiffFile {
        path: path.to_string(),
        old_path: None,
        hunks: vec![hunk],
        status: DiffFileStatus::Untracked,
        is_binary: false,
        too_large_size: None,
    }
}

/// Parse the `b/` path from a `diff --git a/… b/…` line.
///
/// Git's `core.quotepath` (on by default) C-quotes paths containing trigger
/// characters (`"`, `\`, control chars <0x20, bytes ≥0x80). Spaces (0x20)
/// do NOT trigger quoting.
///
/// Two quoting formats exist:
///
/// - **Fully-quoted**: both sides have trigger chars. Git's `quote_two_c_style()`
///   wraps each side in its own double-quote pair:
///   `diff --git "a/file\"name.rs" "b/file\"name.rs"`.
///   The `b/` prefix is INSIDE the quotes — a naive `find(" b/")` fails.
///
/// - **Asymmetrical** (possible for renames where only one path has trigger
///   chars): `diff --git "a/file\"x.rs" b/normal.rs` or
///   `diff --git a/normal.rs "b/file\"x.rs"`.
///
/// Returns `None` if the line is malformed, if escape sequences are invalid,
/// or if the expected `b/` prefix is missing.
///
/// # What this function does NOT handle
///
/// - `---`/`+++` lines: these are only checked for `/dev/null` to detect
///   added/deleted files. They are NOT path-extraction sites even though git
///   may C-quote them. Extracting paths from them would be a bug.
///
/// - `rename to` lines: the `b/` path from `diff --git` is authoritative.
///   `rename to` does not need separate extraction.
///
/// - Combined diff format (`diff --combined` / `diff --cc`): these produce
///   different header formats. Our `git show -m` invocation never produces
///   these, and they are not handled by this parser.
fn parse_diff_git_line(line: &str) -> Option<String> {
    let rest = line.strip_prefix("diff --git ")?;
    let b_part = find_b_part(rest)?;
    extract_b_path(b_part)
}

/// Locate the b-part (second token) in the remainder of a `diff --git` line.
///
/// The a-part may be double-quoted and contain escaped quotes/spaces, so a
/// simple split on the first space is unreliable. For unquoted a-parts,
/// spaces in paths do NOT trigger C-style quoting (per git's `quote_c_style`),
/// and the b-part may itself be quoted. We search for the boundary marker
/// (` b/` for unquoted b-part, ` "b/` for quoted b-part).
fn find_b_part(rest: &str) -> Option<&str> {
    let bytes = rest.as_bytes();

    if bytes.first() == Some(&b'"') {
        // Quoted a-part: scan for the closing unescaped quote,
        // then skip whitespace to reach the b-part.
        let mut i: usize = 1;
        while i < bytes.len() {
            if bytes[i] == b'\\' {
                i += 2; // skip escape sequence (backslash + one char)
            } else if bytes[i] == b'"' {
                i += 1; // skip closing quote
                break;
            } else {
                i += 1;
            }
        }
        while i < bytes.len() && bytes[i] == b' ' {
            i += 1;
        }
        if i >= bytes.len() {
            return None;
        }
        Some(&rest[i..])
    } else {
        // Unquoted a-part. Paths may contain spaces (which don't trigger
        // C-style quoting). The b-part can be either quoted (`"b/…"`) or
        // unquoted (`b/…`). Try the quoted separator first.
        if let Some(idx) = rest.find(" \"b/") {
            Some(&rest[idx + 1..])
        } else {
            rest.find(" b/").map(|idx| &rest[idx + 1..])
        }
    }
}

/// Extract the unescaped path from the b-part token.
///
/// The b-part is either `b/path` (unquoted, no trigger chars) or
/// `"b/path\"with\"escapes"` (quoted, the `b/` prefix is inside the quotes).
fn extract_b_path(b_part: &str) -> Option<String> {
    let path = unquote_c_style(b_part)?;
    Some(path.strip_prefix("b/")?.to_string())
}

/// Parse a single range token like `-10,7` or `+10` from a hunk header.
/// Returns the starting line number, or `None` if the token is malformed.
fn parse_range_token(part: &str, prefix: char) -> Option<usize> {
    let remainder = part.strip_prefix(prefix)?;
    remainder.split(',').next()?.parse::<usize>().ok()
}

/// Parse a hunk header: `@@ -old_start,old_count +new_start,new_count @@ [context]`.
///
/// Returns `(old_start, new_start)`, defaulting to `1` for malformed or missing ranges.
///
/// # Correctness
///
/// The parser must guard against two types of tokens that look like range
/// prefixes but carry different meaning in trailing context:
///
/// - **`->`** (arrow in Rust function signatures): after the closing `@@`,
///   a token like `->` starts with `-` but is not a range token. The parser
///   stops at the first token starting with `@` (the closing delimiter), so
///   trailing-context tokens are never inspected.
/// - **`a + b`** (expressions in context): the `+b` token starts with `+`
///   but appears after the closing `@@` delimiter — again stopped by the
///   `@` break.
///
/// The `@` break alone is sufficient — [`parse_range_token`] simply returns
/// `None` for any non-numeric remainder (its `parse::<usize>()` call fails),
/// so spurious `-` or `+` tokens like `->` are harmlessly skipped.
fn parse_hunk_header(header: &str) -> (usize, usize) {
    // Example inputs:
    //   @@ -10,7 +10,9 @@ fn main() {
    //   @@ -1 +2 @@ fn main()
    //   @@ -0,0 +1,3 @@
    //   @@ -10,7 +10,9 @@ fn process() -> Result<()>

    let after_open = header.trim_start_matches('@');

    let mut old_start = 1;
    let mut new_start = 1;

    for part in after_open.split_whitespace() {
        // Stop when we hit the closing @@ delimiter — any remaining
        // tokens are context and must not influence line numbers.
        if part.starts_with('@') {
            break;
        }

        if part.starts_with('-')
            && let Some(n) = parse_range_token(part, '-')
        {
            old_start = n;
        } else if part.starts_with('+')
            && let Some(n) = parse_range_token(part, '+')
        {
            new_start = n;
        }
    }

    (old_start, new_start)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_diff() {
        let diff = r#"diff --git a/src/main.rs b/src/main.rs
index abc123..def456 100644
--- a/src/main.rs
+++ b/src/main.rs
@@ -1,3 +1,4 @@
 fn main() {
-    println!("hello");
+    println!("hello world");
     println!("goodbye");
 }
"#;
        let output = parse_git_diff(diff);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].path, "src/main.rs");
        assert_eq!(output[0].hunks.len(), 1);
        assert!(output[0].hunks[0].lines.len() >= 3);
        assert_eq!(output[0].hunks[0].lines[0].kind, DiffLineKind::Context);
        assert_eq!(output[0].hunks[0].lines[1].kind, DiffLineKind::Removed);
        assert_eq!(output[0].hunks[0].lines[2].kind, DiffLineKind::Added);
    }

    #[test]
    fn test_new_file() {
        let diff = r#"diff --git a/new.rs b/new.rs
new file mode 100644
index 0000000..abc123
--- /dev/null
+++ b/new.rs
@@ -0,0 +1,2 @@
+fn hello() {
+    println!("new");
+}
"#;
        let output = parse_git_diff(diff);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].status, DiffFileStatus::Added);
        assert!(!output[0].hunks[0].lines.is_empty());
        assert_eq!(output[0].hunks[0].lines[0].new_line_number, Some(1));
    }

    #[test]
    fn test_deleted_file() {
        let diff = r#"diff --git a/old.rs b/old.rs
deleted file mode 100644
index abc123..0000000
--- a/old.rs
+++ /dev/null
@@ -1,2 +0,0 @@
-fn hello() {
-    println!("bye");
-}
"#;
        let output = parse_git_diff(diff);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].status, DiffFileStatus::Deleted);
        assert!(!output[0].hunks[0].lines.is_empty());
        assert_eq!(output[0].hunks[0].lines[0].kind, DiffLineKind::Removed);
    }

    #[test]
    fn test_binary() {
        let diff = r"diff --git a/image.png b/image.png
index abc..def 100644
Binary files a/image.png and b/image.png differ
";
        let output = parse_git_diff(diff);
        assert_eq!(output.len(), 1);
        assert!(output[0].is_binary);
    }

    #[test]
    fn test_empty_diff() {
        let output = parse_git_diff("");
        assert!(output.is_empty());
    }

    // ── parse_diff_git_line tests ─────────────────────────────────────

    #[test]
    fn test_parse_diff_git_line() {
        // Cases: (diff_line, expected_b_path).
        // None-returning cases are not covered — all existing tests
        // exercise only Some-returning paths.
        let cases: &[(&str, Option<&str>)] = &[
            // ── unquoted path ──
            (
                "diff --git a/src/main.rs b/src/main.rs",
                Some("src/main.rs"),
            ),
            // ── fully quoted (both sides have trigger chars) ──
            (
                r#"diff --git "a/file\"name.rs" "b/file\"name.rs""#,
                Some("file\"name.rs"),
            ),
            (
                r#"diff --git "a/path\\with\\backslash.rs" "b/path\\with\\backslash.rs""#,
                Some("path\\with\\backslash.rs"),
            ),
            (
                r#"diff --git "a/file\tname.rs" "b/file\tname.rs""#,
                Some("file\tname.rs"),
            ),
            // Multiple trigger chars: tab, double-quote, backslash.
            (
                r#"diff --git "a/file\t\"quote\"\\n.rs" "b/file\t\"quote\"\\n.rs""#,
                Some("file\t\"quote\"\\n.rs"),
            ),
            // ── asymmetrical (only one side quoted) ──
            (
                r#"diff --git "a/file\"x.rs" b/normal.rs"#,
                Some("normal.rs"),
            ),
            (
                r#"diff --git a/normal.rs "b/file\"x.rs""#,
                Some("file\"x.rs"),
            ),
        ];
        for (i, (input, expected)) in cases.iter().enumerate() {
            let result = parse_diff_git_line(input);
            assert_eq!(
                result.as_deref(),
                *expected,
                "case {i}: parse_diff_git_line({input:?})"
            );
        }
    }

    #[test]
    fn test_rename_from_cases() {
        // Cases: (name, diff, expected_old_path, expected_path).
        let cases: &[(&str, &str, Option<&str>, &str)] = &[
            // Unquoted rename — no special characters.
            (
                "unquoted",
                r"diff --git a/old.rs b/new.rs
similarity index 100%
rename from old.rs
rename to new.rs
",
                Some("old.rs"),
                "new.rs",
            ),
            // Quoted with escape sequences (double-quote in filename).
            (
                "quoted_with_escapes",
                r#"diff --git "a/old\"name.rs" "b/new\"name.rs"
similarity index 100%
rename from "old\"name.rs"
rename to "new\"name.rs"
"#,
                Some("old\"name.rs"),
                "new\"name.rs",
            ),
            // Tab character triggers quoting.
            (
                "quoted_tab_in_name",
                r#"diff --git "a/old\tname.rs" "b/new\tname.rs"
similarity index 100%
rename from "old\tname.rs"
rename to "new\tname.rs"
"#,
                Some("old\tname.rs"),
                "new\tname.rs",
            ),
            // Spaces without quoting — no trigger characters.
            (
                "no_trigger_chars",
                r"diff --git a/old name.rs b/new name.rs
similarity index 100%
rename from old name.rs
rename to new name.rs
",
                Some("old name.rs"),
                "new name.rs",
            ),
        ];
        for (i, (name, diff, expected_old_path, expected_path)) in cases.iter().enumerate() {
            let output = parse_git_diff(diff);
            assert_eq!(output.len(), 1, "case {i} ({name})");
            assert_eq!(
                output[0].status,
                DiffFileStatus::Renamed,
                "case {i} ({name})"
            );
            assert_eq!(
                output[0].old_path.as_deref(),
                *expected_old_path,
                "case {i} ({name}): old_path mismatch"
            );
            assert_eq!(
                output[0].path, *expected_path,
                "case {i} ({name}): path mismatch"
            );
        }
    }

    #[test]
    fn test_untracked_file() {
        let file = make_untracked_diff_file("src/untracked.rs", "fn main() {}");
        assert_eq!(file.status, DiffFileStatus::Untracked);
        assert_eq!(file.hunks[0].lines.len(), 1);
        assert_eq!(file.hunks[0].lines[0].kind, DiffLineKind::Added);
    }

    #[test]
    fn test_multi_hunk_diff() {
        let diff = r"diff --git a/src/lib.rs b/src/lib.rs
index abc123..def456 100644
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1,3 +1,3 @@
 fn top() {
-    old_top();
+    new_top();
 }
@@ -10,3 +10,3 @@
 fn bottom() {
-    old_bottom();
+    new_bottom();
 }
@@ -100,6 +100,6 @@
 fn middle() {
-    old_line1();
-    old_line2();
+    new_line1();
+    new_line2();
     context_line();
 }
";
        let output = parse_git_diff(diff);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].path, "src/lib.rs");
        assert_eq!(output[0].hunks.len(), 3);
        // First hunk
        assert_eq!(output[0].hunks[0].header, "@@ -1,3 +1,3 @@");
        assert_eq!(output[0].hunks[0].lines[1].kind, DiffLineKind::Removed);
        assert_eq!(output[0].hunks[0].lines[2].kind, DiffLineKind::Added);
        // Second hunk
        assert_eq!(output[0].hunks[1].header, "@@ -10,3 +10,3 @@");
        assert_eq!(output[0].hunks[1].lines[1].kind, DiffLineKind::Removed);
        assert_eq!(output[0].hunks[1].lines[2].kind, DiffLineKind::Added);
        // Third hunk — two changes + context
        assert_eq!(output[0].hunks[2].header, "@@ -100,6 +100,6 @@");
        assert_eq!(output[0].hunks[2].lines.len(), 7);
        assert_eq!(output[0].hunks[2].lines[0].kind, DiffLineKind::Context);
        assert_eq!(output[0].hunks[2].lines[1].kind, DiffLineKind::Removed);
        assert_eq!(output[0].hunks[2].lines[2].kind, DiffLineKind::Removed);
        assert_eq!(output[0].hunks[2].lines[3].kind, DiffLineKind::Added);
        assert_eq!(output[0].hunks[2].lines[4].kind, DiffLineKind::Added);
        assert_eq!(output[0].hunks[2].lines[5].kind, DiffLineKind::Context);
        assert_eq!(output[0].hunks[2].lines[6].kind, DiffLineKind::Context);
    }

    #[test]
    fn test_no_newline_annotation_skipped_and_counters_correct() {
        // Files without trailing newlines produce a `\ No newline at end of file`
        // annotation after the last content line. This line is metadata and must
        // NOT increment line counters — otherwise highlight lookups would be
        // off-by-one for all subsequent hunks.
        let diff = r"diff --git a/foo.rs b/foo.rs
index abc123..def456 100644
--- a/foo.rs
+++ b/foo.rs
@@ -1,3 +1,3 @@
 line1
 line2
 line3
\ No newline at end of file
@@ -10,2 +10,2 @@
 other1
 other2
";
        let output = parse_git_diff(diff);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].hunks.len(), 2);
        // First hunk: 3 context lines, no annotation line counted.
        assert_eq!(output[0].hunks[0].lines.len(), 3);
        assert_eq!(output[0].hunks[0].lines[0].old_line_number, Some(1));
        assert_eq!(output[0].hunks[0].lines[2].old_line_number, Some(3));
        // Second hunk: line numbers must pick up at 10, not 11.
        assert_eq!(output[0].hunks[1].lines.len(), 2);
        assert_eq!(output[0].hunks[1].lines[0].old_line_number, Some(10));
        assert_eq!(output[0].hunks[1].lines[1].old_line_number, Some(11));
    }

    // ── parse_hunk_header unit tests ──────────────────────────────────

    #[test]
    fn test_parse_hunk_header_cases() {
        // Cases: (name, hunk_header, expected_old, expected_new).
        // Each name corresponds to the original standalone test for traceability.
        let cases: &[(&str, &str, usize, usize)] = &[
            // Rust function signatures contain `->` which must not corrupt old_start.
            // (The -> token appears after the closing @@ delimiter and is never
            // reached by the parser loop — it breaks on @@ first.)
            (
                "hunk_context_with_arrow",
                "@@ -10,7 +10,9 @@ fn process() -> Result<()>",
                10,
                10,
            ),
            // Expressions like `a + b` in context must not corrupt new_start.
            // (Any `->` token is after the closing @@ delimiter — the parser
            // breaks on @@ first, so spurious tokens are not inspected.)
            (
                "hunk_context_with_plus",
                "@@ -5,3 +5,4 @@ fn add(a: i32, b: i32) -> i32 { let x = a + b; }",
                5,
                5,
            ),
            // A context that produces `@@` as a whitespace-delimited token
            // must not cause the parser to consume tokens after the real delimiter.
            ("hunk_at_at_in_context", "@@ -1,3 +1,3 @@ @@ -this", 1, 1),
            // Standard hunk header with plain context.
            (
                "hunk_plain_context",
                "@@ -100,6 +200,7 @@ fn main() {",
                100,
                200,
            ),
            // No context string after the closing @@.
            ("hunk_no_context", "@@ -0,0 +1,5 @@", 0, 1),
            // Single-line hunk without comma-count on old range: `@@ -old_start +new_start @@`.
            ("hunk_no_count_first", "@@ -1 +1 @@ fn single_line()", 1, 1),
            // Single-line hunk without comma-count — different values.
            ("hunk_no_count_second", "@@ -5 +3 @@ fn another()", 5, 3),
            // Context like `if x < -1` produces a `-1` token — must be skipped.
            // The remainder after stripping `-` is `"1"`, which is all digits,
            // so digit validation alone would not reject it. Only the @-break
            // (stopping at the closing @@ delimiter) protects against this.
            // We test a constructed header where `-1` appears after the real
            // range tokens — it must be skipped via the @-break.
            (
                "hunk_negative_number_in_context",
                "@@ -3,2 +3,2 @@ fn check() { if x < -1 { } }",
                3,
                3,
            ),
        ];
        for (i, (name, input, expected_old, expected_new)) in cases.iter().enumerate() {
            let (old, new) = parse_hunk_header(input);
            assert_eq!(
                old, *expected_old,
                "case {i} ({name}): old_start mismatch. Input: {input:?}"
            );
            assert_eq!(
                new, *expected_new,
                "case {i} ({name}): new_start mismatch. Input: {input:?}"
            );
        }
    }
}
