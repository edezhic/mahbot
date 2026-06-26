//! Git diff parsing — parse unified diff output into structured data.
//!
//! Handles `git diff HEAD --no-color --find-renames` output plus
//! untracked files from `git status --porcelain`.

use std::collections::HashSet;
use std::path::Path;
use tracing::warn;

/// Result of a successful `git commit` — the full hash and line stats.
#[derive(Debug, Clone)]
pub struct CommitInfo {
    /// Full 40-character commit SHA from `git rev-parse HEAD`.
    pub hash: String,
    /// Total lines added across all changed files.
    pub lines_added: i64,
    /// Total lines removed across all changed files.
    pub lines_removed: i64,
}

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

/// Flush the current file (with optional pending hunk) into the files vec.
fn flush_file(
    current_file: &mut Option<DiffFile>,
    current_hunk: &mut Option<DiffHunk>,
    files: &mut Vec<DiffFile>,
) {
    if let Some(mut file) = current_file.take() {
        if let Some(hunk) = current_hunk.take() {
            file.hunks.push(hunk);
        }
        files.push(file);
    }
}

/// Handle a `diff --git` line: flush previous file, reset counters, and create a new file entry.
fn handle_diff_git_header(
    line: &str,
    current_file: &mut Option<DiffFile>,
    current_hunk: &mut Option<DiffHunk>,
    files: &mut Vec<DiffFile>,
    old_counter: &mut usize,
    new_counter: &mut usize,
) {
    flush_file(current_file, current_hunk, files);
    *old_counter = 0;
    *new_counter = 0;

    if let Some(path) = parse_diff_git_line(line) {
        *current_file = Some(DiffFile {
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
fn handle_rename_from(line: &str, current_file: &mut Option<DiffFile>) {
    let Some(f) = current_file.as_mut() else {
        return;
    };

    f.status = DiffFileStatus::Renamed;
    let raw = line.strip_prefix("rename from ").unwrap_or("");
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
fn handle_hunk_header(
    line: &str,
    current_file: &mut Option<DiffFile>,
    current_hunk: &mut Option<DiffHunk>,
    old_counter: &mut usize,
    new_counter: &mut usize,
) {
    if let Some(hunk) = current_hunk.take()
        && let Some(f) = current_file
    {
        f.hunks.push(hunk);
    }
    let (old_start, new_start) = parse_hunk_header(line);
    *old_counter = old_start;
    *new_counter = new_start;
    *current_hunk = Some(DiffHunk {
        header: line.to_string(),
        lines: Vec::new(),
    });
}

/// Handle a content line within a hunk: classify as Added/Removed/Context, track
/// line numbers, and push to the hunk.
///
/// Returns `true` if the line is an annotation (`\ No newline at end of file` or
/// an unknown line) that should be skipped — the caller should `continue`.
fn handle_diff_content_line(
    line: &str,
    hunk: &mut DiffHunk,
    old_counter: &mut usize,
    new_counter: &mut usize,
) {
    let line_kind = if line.starts_with('+') {
        DiffLineKind::Added
    } else if line.starts_with('-') {
        DiffLineKind::Removed
    } else if line.starts_with(' ') {
        DiffLineKind::Context
    } else if line == r"\ No newline at end of file" {
        // Skip this annotation — handled implicitly
        return;
    } else {
        // Unknown line — skip
        return;
    };

    let content = line[1..].trim_end_matches('\r');
    let (old_num, new_num) = match line_kind {
        DiffLineKind::Added => {
            let n = Some(*new_counter);
            *new_counter += 1;
            (None, n)
        }
        DiffLineKind::Removed => {
            let n = Some(*old_counter);
            *old_counter += 1;
            (n, None)
        }
        DiffLineKind::Context => {
            let o = Some(*old_counter);
            let n = Some(*new_counter);
            *old_counter += 1;
            *new_counter += 1;
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

/// Parse `git diff HEAD --no-color --find-renames` output.
#[must_use]
pub fn parse_git_diff(diff_output: &str) -> Vec<DiffFile> {
    let mut files: Vec<DiffFile> = Vec::new();
    let mut current_file: Option<DiffFile> = None;
    let mut current_hunk: Option<DiffHunk> = None;

    // Track old/new line counters for line number assignment
    let mut old_counter: usize = 0;
    let mut new_counter: usize = 0;

    for line in diff_output.lines() {
        if line.starts_with("diff --git ") {
            handle_diff_git_header(
                line,
                &mut current_file,
                &mut current_hunk,
                &mut files,
                &mut old_counter,
                &mut new_counter,
            );
        } else if line.starts_with("index ")
            || line.starts_with("new file mode ")
            || line.starts_with("deleted file mode ")
            || line.starts_with("old mode ")
            || line.starts_with("new mode ")
        {
            // Metadata lines — currently just skip
        } else if line.starts_with("--- ") || line.starts_with("+++ ") {
            // File markers
            if let Some(ref mut f) = current_file {
                if line.starts_with("--- /dev/null") && f.status != DiffFileStatus::Renamed {
                    f.status = DiffFileStatus::Added;
                } else if line.starts_with("+++ /dev/null") && f.status != DiffFileStatus::Renamed {
                    f.status = DiffFileStatus::Deleted;
                }
            }
        } else if line.starts_with("rename from ") {
            handle_rename_from(line, &mut current_file);
        } else if line.starts_with("rename to ") {
            // The path is already captured from diff --git; status already set at rename from.
        } else if line.starts_with("Binary files ") {
            if let Some(ref mut f) = current_file {
                f.is_binary = true;
            }
        } else if line.starts_with("@@") {
            handle_hunk_header(
                line,
                &mut current_file,
                &mut current_hunk,
                &mut old_counter,
                &mut new_counter,
            );
        } else if let Some(hunk) = &mut current_hunk {
            handle_diff_content_line(line, hunk, &mut old_counter, &mut new_counter);
        }
    }

    // Flush final file + hunk
    flush_file(&mut current_file, &mut current_hunk, &mut files);

    files
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

/// Strip surrounding double-quotes and unescape C-style escapes.
///
/// If the input starts with `"` and ends with `"`, strips the quotes and
/// calls [`unescape_c_style`] on the inner content. Otherwise returns the
/// input as-is (no unescaping needed — git only C-quotes paths that contain
/// trigger characters).
///
/// This is the standard pattern for handling git's quoted path output.
/// Compare with git's own `unquote_c_style` which performs the same
/// quote-strip-then-unescape logic.
#[must_use]
pub fn unquote_c_style(raw: &str) -> Option<String> {
    if let Some(inner) = raw.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        unescape_c_style(inner)
    } else {
        Some(raw.to_string())
    }
}

/// Unescape C-style escape sequences from a git path name.
///
/// Supports the same escapes as git's `unquote_c_style`:
/// - `\"` → literal `"`, `\\` → literal `\`
/// - `\t` → tab, `\n` → newline, `\a` → bell, `\b` → backspace
/// - `\f` → form feed, `\r` → carriage return, `\v` → vertical tab
/// - `\0`–`\3` followed by 1–3 octal digits → byte value
///
/// Malformed escapes cause this function to return `None`:
/// - `\` at end of string (dangling backslash)
/// - `\x` or any other unrecognized escape letter
/// - `\4`–`\7` followed by a digit (git rejects these as invalid octal prefixes)
///
/// Non-UTF-8 bytes produced by octal escapes are handled via
/// `String::from_utf8_lossy` — pragmatic for macOS where non-UTF-8 paths
/// are filesystem-impossible.
pub fn unescape_c_style(input: &str) -> Option<String> {
    let mut result = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i: usize = 0;

    while i < bytes.len() {
        if bytes[i] == b'\\' {
            i += 1; // consume backslash
            if i >= bytes.len() {
                warn!(
                    input = %input,
                    "unescape_c_style: dangling backslash at end of string"
                );
                return None;
            }
            match bytes[i] {
                b'"' => result.push('"'),
                b'\\' => result.push('\\'),
                b't' => result.push('\t'),
                b'n' => result.push('\n'),
                b'a' => result.push('\x07'),
                b'b' => result.push('\x08'),
                b'f' => result.push('\x0c'),
                b'r' => result.push('\r'),
                b'v' => result.push('\x0b'),
                b'0'..=b'3' => {
                    // Octal escape: 1–3 octal digits.
                    let digits_start = i;
                    i += 1;
                    let mut digit_count = 1;
                    while digit_count < 3 && i < bytes.len() && bytes[i].is_ascii_digit() {
                        if !(b'0'..=b'7').contains(&bytes[i]) {
                            break;
                        }
                        i += 1;
                        digit_count += 1;
                    }
                    let octal_str = std::str::from_utf8(&bytes[digits_start..i]).ok()?;
                    let Ok(byte_val) = u8::from_str_radix(octal_str, 8) else {
                        warn!(
                            input = %input, octal = %octal_str,
                            "unescape_c_style: invalid octal escape"
                        );
                        return None;
                    };
                    result.push_str(&String::from_utf8_lossy(&[byte_val]));
                    continue; // skip the i += 1 at end of loop
                }
                b'4'..=b'7' => {
                    // \4–\7 are not valid octal prefixes in git's unquote_c_style.
                    // If followed by a digit, it's a malformed octal attempt.
                    if i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
                        warn!(
                            input = %input,
                            ch = %(bytes[i] as char),
                            "unescape_c_style: invalid octal prefix \\4–\\7 followed by digit"
                        );
                        return None;
                    }
                    // Otherwise: literal digit (backslash consumed, no special meaning).
                    result.push(bytes[i] as char);
                }
                _ => {
                    warn!(
                        input = %input,
                        ch = %(bytes[i] as char),
                        "unescape_c_style: unrecognized escape sequence"
                    );
                    return None;
                }
            }
            i += 1;
        } else {
            result.push(bytes[i] as char);
            i += 1;
        }
    }

    Some(result)
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
    // Locate the b-part of the two space-separated tokens.
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
        // Skip whitespace between the two parts.
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
/// `header` is the original hunk header string, used for diagnostic messages.
fn parse_range_token(part: &str, prefix: char, header: &str) -> Option<usize> {
    let remainder = part.strip_prefix(prefix)?;
    if remainder.is_empty() || !remainder.chars().all(|c| c.is_ascii_digit() || c == ',') {
        warn!("Skipping non-numeric '{prefix}' token {part:?} in hunk header {header:?}");
        return None;
    }
    remainder
        .split(',')
        .next()
        .and_then(|s| s.parse::<usize>().ok())
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
/// The character validation in [`parse_range_token`] provides a second
/// line of defense: even if a spurious `-` or `+` token somehow appeared
/// before the closing `@@`, non-digit/non-comma characters cause it to be
/// skipped (e.g. `->` yields remainder `>` which fails validation).
fn parse_hunk_header(header: &str) -> (usize, usize) {
    // Example inputs:
    //   @@ -10,7 +10,9 @@ fn main() {
    //   @@ -1 +2 @@ fn main()
    //   @@ -0,0 +1,3 @@
    //   @@ -10,7 +10,9 @@ fn process() -> Result<()>

    // Strip the opening @@ delimiter.
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
            && let Some(n) = parse_range_token(part, '-', header)
        {
            old_start = n;
        } else if part.starts_with('+')
            && let Some(n) = parse_range_token(part, '+', header)
        {
            new_start = n;
        }
    }

    (old_start, new_start)
}

/// Check if a directory contains a `.git` entry (file for worktrees, directory otherwise).
#[must_use]
pub fn is_git_repo(path: &Path) -> bool {
    path.join(".git").exists()
}

/// Run `git diff HEAD --no-color --find-renames` (when `commit_ref` is `None`)
/// or `git show -m <commit_ref> --no-color --find-renames --format=""` (when `Some`).
///
/// `-m` splits merge commits into per-parent diffs, producing standard unified diff
/// blocks that `parse_git_diff` can handle. This may produce duplicate file paths in the
/// tree sidebar (one `diff --git` block per parent), which is expected — each parent
/// comparison is a different diff.
/// `--format=""` suppresses the commit log header.
pub async fn run_git_diff(repo_path: &Path, commit_ref: Option<&str>) -> Result<String, String> {
    if let Some(hash) = commit_ref {
        run_git_command(
            repo_path,
            &[
                "show",
                "-m",
                hash,
                "--no-color",
                "--find-renames",
                "--format=",
            ],
        )
        .await
    } else {
        run_git_command(repo_path, &["diff", "HEAD", "--no-color", "--find-renames"]).await
    }
}

/// Run `git status --porcelain` and return the output.
pub async fn run_git_status(repo_path: &Path) -> Result<String, String> {
    run_git_command(repo_path, &["status", "--porcelain"]).await
}

/// Run `git show HEAD:<path>` (when `commit_ref` is `None`)
/// or `git show <commit_ref>:<path>` (when `Some`) and return the file content.
///
/// Returns `Ok(None)` if the file does not exist at that ref (new/untracked files,
/// or root-commit `~1` which has no parent).
///
/// **`~1` parent refs**: The caller constructs the parent hash. To get the parent
/// version, pass `commit_ref = Some(&format!("{hash}~1"))`.
pub async fn run_git_show(
    repo_path: &Path,
    file_path: &str,
    commit_ref: Option<&str>,
) -> Result<Option<String>, String> {
    let show_arg = if let Some(hash) = commit_ref {
        format!("{hash}:{file_path}")
    } else {
        format!("HEAD:{file_path}")
    };
    match run_git_command(repo_path, &["show", &show_arg]).await {
        Ok(output) => Ok(Some(output)),
        Err(_) => Ok(None),
    }
}
/// Run a git command and return stdout as string.
pub async fn run_git_command(repo_path: &Path, args: &[&str]) -> Result<String, String> {
    let output = tokio::process::Command::new("git")
        .args(args)
        .current_dir(repo_path)
        .env("LC_ALL", "C")
        .output()
        .await
        .map_err(|e| format!("Failed to run git: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(format!("Git command failed: {stderr}"));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Run `git check-ignore --stdin` with the given file paths.
/// Pipes paths via stdin and returns the set of paths that are ignored.
///
/// Exit code 0 means some paths matched (output lists them, one per line).
/// Exit code 1 means no paths were ignored (not a failure — returns empty set).
/// Any other exit code is treated as an error.
pub async fn run_git_check_ignore(
    repo_path: &Path,
    paths: &[String],
) -> Result<HashSet<String>, String> {
    use std::process::Stdio;

    let mut child = tokio::process::Command::new("git")
        .args(["check-ignore", "--stdin"])
        .current_dir(repo_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn git check-ignore: {e}"))?;

    // Write all paths to stdin, then close it.
    let mut stdin = child.stdin.take().expect("stdin not captured");
    for path in paths {
        use tokio::io::AsyncWriteExt;
        stdin
            .write_all(path.as_bytes())
            .await
            .map_err(|e| format!("Failed to write to git stdin: {e}"))?;
        stdin
            .write_all(b"\n")
            .await
            .map_err(|e| format!("Failed to write newline to git stdin: {e}"))?;
    }
    drop(stdin);

    let output = child
        .wait_with_output()
        .await
        .map_err(|e| format!("Failed to wait for git check-ignore: {e}"))?;

    // Exit code 1 means "no files ignored" — not a failure, return empty set.
    if output.status.code() == Some(1) {
        return Ok(HashSet::new());
    }

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(format!("Git check-ignore failed: {stderr}"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let ignored: HashSet<String> = stdout.lines().map(ToString::to_string).collect();
    Ok(ignored)
}

/// Stage all changes and commit with the given message.
/// Runs `git add -A` followed by `git commit -m "<msg>"`,
/// then captures the full SHA via `git rev-parse HEAD` and
/// line stats via `git diff --numstat`.
pub async fn run_git_commit(repo_path: &Path, message: &str) -> Result<CommitInfo, String> {
    // Stage all changes (tracked, untracked, removed) in the worktree.
    run_git_command(repo_path, &["add", "-A"]).await?;

    run_git_command(repo_path, &["commit", "-m", message])
        .await
        .map_err(|e| format!("Commit failed: {}", e.trim()))?;

    // Capture the full 40-char SHA — reliable source, not abbreviated.
    let hash = match run_git_command(repo_path, &["rev-parse", "HEAD"]).await {
        Ok(out) => out.trim().to_string(),
        Err(e) => {
            warn!(
                error = %e,
                "git rev-parse HEAD failed after successful commit — commit exists, returning unknown hash"
            );
            return Ok(CommitInfo {
                hash: "unknown".into(),
                lines_added: 0,
                lines_removed: 0,
            });
        }
    };

    // Capture line stats via --numstat. Try HEAD~1..HEAD first.
    if let Some((lines_added, lines_removed)) =
        parse_numstat(repo_path, &["diff", "--numstat", "HEAD~1..HEAD"]).await
    {
        Ok(CommitInfo {
            hash,
            lines_added,
            lines_removed,
        })
    } else {
        // HEAD~1 doesn't exist (first commit) — fall back to the empty tree hash.
        let (lines_added, lines_removed) = parse_numstat(
            repo_path,
            &[
                "diff",
                "--numstat",
                "4b825dc642cb6eb9a060e54bf899dcee6a7b9e2a",
                "HEAD",
            ],
        )
        .await
        .unwrap_or((0, 0));
        Ok(CommitInfo {
            hash,
            lines_added,
            lines_removed,
        })
    }
}

/// Run `git diff --numstat <args...>` and sum the line stats across all files.
///
/// Returns `Some((lines_added, lines_removed))` on success (even if the diff
/// is empty). Returns `None` on any error — command failure, non-zero exit,
/// or spawn failure — after logging a warning. Stats are non-critical.
async fn parse_numstat(repo_path: &Path, args: &[&str]) -> Option<(i64, i64)> {
    let stdout = match run_git_command(repo_path, args).await {
        Ok(out) => out,
        Err(e) => {
            warn!(args = ?args, error = %e, "git diff --numstat failed");
            return None;
        }
    };

    let mut lines_added: i64 = 0;
    let mut lines_removed: i64 = 0;

    for line in stdout.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 2 {
            lines_added += parts[0].parse::<i64>().unwrap_or(0);
            lines_removed += parts[1].parse::<i64>().unwrap_or(0);
        }
    }

    Some((lines_added, lines_removed))
}

/// Check if git is installed.
pub async fn git_is_installed() -> bool {
    tokio::process::Command::new("git")
        .arg("--version")
        .output()
        .await
        .is_ok_and(|o| o.status.success())
}

/// Check if a git repo has any commits.
pub async fn git_has_commits(repo_path: &Path) -> Result<bool, String> {
    let output = tokio::process::Command::new("git")
        .args(["rev-list", "-n", "1", "HEAD"])
        .current_dir(repo_path)
        .output()
        .await
        .map_err(|e| format!("Failed to run git: {e}"))?;

    // If the command fails, there are no commits or something is wrong
    if !output.status.success() {
        return Ok(false);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(!stdout.trim().is_empty())
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

    // ── unescape_c_style tests ────────────────────────────────────────

    #[test]
    fn test_unescape_c_style() {
        // Cases: (input, expected_output).
        // Uses Option<&str> — compared via .as_deref() against the
        // function's Option<String> return type.
        let cases: &[(&str, Option<&str>)] = &[
            // ── basic escapes ──
            (
                r#"hello\"world\\test\nline\there"#,
                Some("hello\"world\\test\nline\there"),
            ),
            // Bell, backspace, formfeed, CR, vertical tab.
            (r"\a\b\f\r\v", Some("\x07\x08\x0c\r\x0b")),
            // ── octal escapes, 1–3 digits ──
            // \0 → NUL (0x00), \1 → SOH (0x01)
            (r"\0\1", Some("\0\x01")),
            // \12 → newline (0x0a), \37 → unit separator (0x1f)
            (r"\12\37", Some("\n\x1f")),
            // \101 → 'A' (0x41), \377 → 0xff → U+FFFD (from_utf8_lossy replacement)
            (r"\101\377", Some("A\u{FFFD}")),
            // Octal stops at non-octal-digit: \12x → newline + 'x'
            (r"\12x", Some("\nx")),
            // \18 → \1 (SOH, 0x01) then '8' (8 is not an octal digit)
            (r"\18", Some("\x018")),
            // ── no-op cases (no escape sequences) ──
            ("plain/path.rs", Some("plain/path.rs")),
            ("", Some("")),
            // ── error cases (return None) ──
            // Dangling backslash at end of string.
            (r"path\", None),
            // \x looks like a hex escape prefix but git's unquote_c_style rejects it.
            (r"\x", None),
            // \q is not a recognized escape sequence.
            (r"\q", None),
            // \40 — \4 followed by digit (git rejects as invalid octal prefix).
            (r"\40", None),
            // \77 — \7 followed by digit (git rejects as invalid octal prefix).
            (r"\77", None),
            // \70 — \7 followed by digit (git rejects as invalid octal prefix).
            (r"\70", None),
            // ── literal-digit cases (\4/\7 not followed by digit) ──
            // \4 at end of string → literal '4'.
            (r"\4", Some("4")),
            // \7 followed by non-digit → literal '7' then 'x'.
            (r"\7x", Some("7x")),
        ];
        for (i, (input, expected)) in cases.iter().enumerate() {
            let result = unescape_c_style(input);
            assert_eq!(
                result.as_deref(),
                *expected,
                "case {i}: unescape_c_style({input:?})"
            );
        }
    }

    // ── rename from integration tests ──────────────────────────────────

    #[test]
    fn test_rename_from_unquoted() {
        let diff = r"diff --git a/old.rs b/new.rs
similarity index 100%
rename from old.rs
rename to new.rs
";
        let output = parse_git_diff(diff);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].status, DiffFileStatus::Renamed);
        assert_eq!(output[0].old_path, Some("old.rs".to_string()));
        assert_eq!(output[0].path, "new.rs");
    }

    #[test]
    fn test_rename_from_quoted_with_escapes() {
        // Trigger char: double-quote in the old filename.
        let diff = r#"diff --git "a/old\"name.rs" "b/new\"name.rs"
similarity index 100%
rename from "old\"name.rs"
rename to "new\"name.rs"
"#;
        let output = parse_git_diff(diff);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].status, DiffFileStatus::Renamed);
        assert_eq!(output[0].old_path, Some("old\"name.rs".to_string()));
        assert_eq!(output[0].path, "new\"name.rs");
    }

    #[test]
    fn test_rename_from_quoted_tab_in_name() {
        // Trigger char: tab.
        let diff = "diff --git \"a/old\\tname.rs\" \"b/new\\tname.rs\"\n\
similarity index 100%\n\
rename from \"old\\tname.rs\"\n\
rename to \"new\\tname.rs\"\n";
        let output = parse_git_diff(diff);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].status, DiffFileStatus::Renamed);
        assert_eq!(output[0].old_path, Some("old\tname.rs".to_string()));
    }

    #[test]
    fn test_rename_from_no_trigger_chars() {
        // Spaces do NOT trigger quoting in git.
        let diff = r"diff --git a/old name.rs b/new name.rs
similarity index 100%
rename from old name.rs
rename to new name.rs
";
        let output = parse_git_diff(diff);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].status, DiffFileStatus::Renamed);
        // Spaces are literal — no quoting means no unquoting needed.
        assert_eq!(output[0].old_path, Some("old name.rs".to_string()));
        assert_eq!(output[0].path, "new name.rs");
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
            // breaks on @@ first, so the warn!() path is not reached here.)
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
