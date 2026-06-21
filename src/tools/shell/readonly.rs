//! Read-only shell command validation.
//!
//! [`check_command`] validates shell commands against a set of rules that
//! distinguish safe inspection commands from workspace-mutating ones.
//! Used by [`crate::tools::shell::ShellTool`] when operating in [`ShellMode::ReadOnly`].

use std::path::Path;

/// Shell execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellMode {
    /// Full shell access — all commands allowed.
    Full,
    /// Read-only shell — only inspection commands allowed.
    ReadOnly,
}

// ── Const tables ─────────────────────────────────────────────────────────

/// Commands rejected unconditionally (any invocation).
///
/// NOTE: Script interpreters (bash, python, node, etc.) and container tools
/// (docker, kubectl, etc.) are intentionally NOT in this list. They are
/// general-purpose tools commonly used for read-only inspection (e.g.,
/// `python --version`, `docker ps`, `kubectl get pods`). Shell prefix
/// stripping covers dangerous wrapper patterns (sudo, eval, exec). The
/// trade-off accepts false negatives through indirection (e.g.,
/// `sh -c "rm -rf /"`, `python3 -c "__import__('os').system('rm -rf /')"`)
/// in favor of not breaking legitimate read-only usage.
const MUTATING_COMMANDS: &[&str] = &[
    // ── File mutation ──
    "rm",
    "rmdir",
    "unlink",
    "shred",
    "cp",
    "mv",
    "touch",
    "mkdir",
    "mkfifo",
    "mknod",
    "ln",
    "install",
    "truncate",
    "fallocate",
    "tee",
    "split",
    "csplit",
    "patch",
    "scp",
    "sftp",
    "chmod",
    "chown",
    "chattr",
    "chflags",
    "setfacl",
    "rsync",
    "zip",
    "unzip",
    "vim",
    "vi",
    "nvim",
    "nano",
    "pico",
    "emacs",
    "ed",
    "code",
    "gedit",
    "sponge",
    "kill",
    "pkill",
    "killall",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "make",
    "cmake",
    "wget",
    "gzip",
    "gunzip",
    "bzip2",
    "xz",
    "zstd",
    // ── Package managers ──
    "npm",
    "yarn",
    "pnpm",
    "pip",
    "pip3",
    "pipenv",
    "poetry",
    "brew",
    "port",
];

/// Safe git subcommands (read-only inspection).
///
/// Note: "stash list" is safe but intentionally excluded from this list — it is
/// fast-tracked in `check_git_segment` via a string-contains early return before
/// the const array is consulted.
const GIT_SAFE_SUBCOMMANDS: &[&str] = &[
    "status",
    "log",
    "diff",
    "show",
    "blame",
    "annotate",
    "shortlog",
    "describe",
    "ls-files",
    "ls-tree",
    "rev-parse",
    "rev-list",
    "for-each-ref",
    "grep",
    "help",
    "version",
    "name-rev",
    "count-objects",
    "verify-pack",
    "verify-commit",
    "verify-tag",
    "check-attr",
    "check-ignore",
    "check-mailmap",
    "check-ref-format",
    "cat-file",
    "cherry",
    "diff-files",
    "diff-index",
    "diff-tree",
    "fmt-merge-msg",
    "fsck",
    "merge-base",
    "merge-file",
    "merge-tree",
    "whatchanged",
    "reflog",
    "range-diff",
    "request-pull",
    "worktree list",
    "config --list",
    "config --get",
    "config --get-all",
    "hash-object",
    "mktag",
    "mktree",
    "stripspace",
    "remote",
    "branch",
    "tag",
];

/// Safe cargo subcommands.
const CARGO_SAFE_SUBCOMMANDS: &[&str] = &[
    "build",
    "check",
    "test",
    "clippy",
    "rustc",
    "metadata",
    "tree",
    "locate-project",
    "pkgid",
    "report",
    "search",
    "info",
    "clean",
    "doc",
    "fmt",
    "generate-lockfile",
    "update",
    "version",
    "verify-project",
    "read-manifest",
    "help",
    "bench",
];

// ── Redirect detection ───────────────────────────────────────────────────

/// Remove heredoc bodies so redirect operators inside them are not scanned.
fn strip_heredoc_bodies(command: &str) -> String {
    let mut out = String::new();
    let mut i = 0;
    let chars: Vec<(usize, char)> = command.char_indices().collect();

    while i < chars.len() {
        if i + 1 < chars.len() && chars[i].1 == '<' && chars[i + 1].1 == '<' {
            out.push(' ');
            i += 2;
            // Optional <<- (strip leading tabs from delimiter line)
            while i < chars.len() && chars[i].1.is_whitespace() {
                i += 1;
            }
            if i < chars.len() && chars[i].1 == '-' {
                i += 1;
            }
            while i < chars.len() && chars[i].1.is_whitespace() {
                i += 1;
            }

            let (delimiter, delim_end) = parse_heredoc_delimiter(command, chars[i].0);
            i = chars
                .iter()
                .position(|(byte, _)| *byte >= delim_end)
                .unwrap_or(chars.len());

            // Skip rest of delimiter line
            while i < chars.len() && chars[i].1 != '\n' {
                i += 1;
            }
            if i < chars.len() {
                i += 1;
            }

            // Skip body until a line equals the delimiter
            let delim_bytes = delimiter.as_bytes();
            while i < chars.len() {
                let line_start = chars[i].0;
                if command[line_start..].starts_with(&delimiter)
                    && (command.len() == line_start + delimiter.len()
                        || matches!(
                            command.as_bytes().get(line_start + delimiter.len()),
                            Some(b'\n' | b'\r')
                        ))
                {
                    i = chars
                        .iter()
                        .position(|(byte, _)| *byte > line_start + delimiter.len())
                        .unwrap_or(chars.len());
                    while i < chars.len() && chars[i].1 != '\n' {
                        i += 1;
                    }
                    if i < chars.len() {
                        i += 1;
                    }
                    break;
                }
                while i < chars.len() && chars[i].1 != '\n' {
                    i += 1;
                }
                if i < chars.len() {
                    i += 1;
                }
            }
            let _ = delim_bytes; // delimiter compared via starts_with above
            continue;
        }

        out.push(chars[i].1);
        i += 1;
    }

    out
}

/// Parse a heredoc delimiter token starting at `start` (byte index).
fn parse_heredoc_delimiter(command: &str, start: usize) -> (String, usize) {
    let rest = &command[start..];
    if let Some(rest) = rest.strip_prefix('\'') {
        if let Some(end) = rest.find('\'') {
            let delim = &rest[..end];
            return (delim.to_string(), start + 1 + end + 1);
        }
    } else if let Some(rest) = rest.strip_prefix('"')
        && let Some(end) = rest.find('"')
    {
        let delim = &rest[..end];
        return (delim.to_string(), start + 1 + end + 1);
    }

    let end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
    (rest[..end].to_string(), start + end)
}

/// Detect output redirect operators in a command string, respecting quote state.
/// Returns true if the command contains a redirect that writes to a
/// non-allowed destination (not `/dev/null`, not temp dir).
fn has_disallowed_redirect(command_str: &str) -> bool {
    let scan_str = strip_heredoc_bodies(command_str);
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;

    // Use `char_indices()` for byte-accurate slicing of the original string
    // and to fix the pre-existing multi-byte UTF-8 bug (the old `bytes[i] as
    // char` approach produced garbage for non-ASCII characters).
    //
    // MUST use a `while let` loop — a `for` loop over the iterator is NOT
    // suitable because multi-character redirect operators (>>, >|, >&, 2>,
    // 2>&1, 1>&2) require manual iterator advancement to skip already-matched
    // chars.  A `for` loop would double-count those chars on the next
    // iteration.
    let mut chars = scan_str.char_indices();

    while let Some((i, c)) = chars.next() {
        // Handle escape tracking locally — backslash escaping is independent
        // of the quote state machine.  The `!in_single` guard ensures that
        // backslashes inside single quotes are literal (matching real shell
        // behavior).
        //
        // # Known limitation
        //
        // Inside double quotes, `\` should only escape `\`, `$`, `` ` ``,
        // `"`, and newline in a real shell.  This code treats any backslash
        // inside double quotes as an escape, which is safe for redirect
        // detection: a quoted redirect operator is harmless, and an escaped
        // actual redirect would be a false negative (allow), also harmless.
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' && !in_single {
            escaped = true;
            continue;
        }

        if !super::check_outside_quotes(c, &mut in_single, &mut in_double) {
            continue;
        }

        // Check for 2>&1 and 1>&2 — pure stderr-to-stdout merges, always
        // allowed.  These are 4-character patterns; after matching we skip
        // the remaining 3 chars with `nth(2)`.
        if scan_str[i..].starts_with("2>&1") || scan_str[i..].starts_with("1>&2") {
            chars.nth(2);
            continue;
        }

        // 2-character redirect operators
        let redirect_len = if scan_str[i..].starts_with(">&")
            || scan_str[i..].starts_with(">>")
            || scan_str[i..].starts_with(">|")
            || scan_str[i..].starts_with("2>")
        {
            2
        } else if c == '>' {
            1
        } else {
            continue;
        };

        // Skip remaining chars of the redirect operator (first char already
        // consumed by the `while let`).  For a 2-char operator, skip 1 more.
        if redirect_len > 1 {
            chars.next();
        }

        // Extract target after redirect operator
        let after = &scan_str[i + redirect_len..].trim_start();
        let target = after
            .split(|ch: char| ch.is_whitespace() || ch == '&' || ch == ';' || ch == '|')
            .next()
            .unwrap_or("");

        if target.is_empty() {
            // No target — bare redirect, reject
            return true;
        }

        // Allowed targets
        if target == "/dev/null" {
            continue;
        }

        let target_path = Path::new(target);
        if target_path.is_absolute() {
            if crate::tools::is_path_under_allowed_temp(target_path) {
                continue;
            }
            // Absolute non-temp non-devnull = disallowed
            return true;
        }

        // Relative redirect to workspace = disallowed
        return true;
    }

    false
}

// ── Main validation function ──────────────────────────────────────────────

/// Validate a shell command for read-only execution.
///
/// Splits chained commands into segments, checks each segment against
/// the allowlists and rejection rules, and returns `Ok(())` if the
/// command is safe, or `Err(String)` with a descriptive rejection message.
pub(super) fn check_command(command_str: &str) -> Result<(), String> {
    let trimmed = command_str.trim();
    if trimmed.is_empty() {
        return Ok(());
    }

    // Redirect detection runs on the full command string to avoid
    // segment splitting breaking compound operators like `>|`.
    if has_disallowed_redirect(trimmed) {
        return Err(format!(
            "⚠️ Read-only mode: command contains a disallowed output redirect.\n\
             Command: `{trimmed}`\n\
             Redirects are only allowed to /dev/null, 2>&1, 1>&2, or paths under /tmp, /var/tmp, or the OS temp directory.\n\
             Suggestion: pipe to a pager (e.g., `| less`) or use `| head` to limit output."
        ));
    }

    let segments = super::extract_command_segments(trimmed);
    for segment in &segments {
        check_segment(segment)?;
    }

    Ok(())
}

/// Construct a read-only rejection error with consistent formatting.
fn reject(cmd: &str, why: &str, suggestion: &str) -> Result<(), String> {
    Err(format!(
        "⚠️ Read-only mode: {why}\n\
         Command: `{cmd}`\n\
         Suggestion: {suggestion}"
    ))
}

/// Scratch-file mutators allowed when all explicit path args are under temp.
const SCRATCH_MUTATORS: &[&str] = &["tee", "touch", "mkdir"];

/// Non-flag arguments from a command segment (after canonicalization).
fn non_flag_path_args(segment: &str) -> Vec<String> {
    let canonical = super::canonical_command(segment);
    let parts: Vec<&str> = canonical.split_whitespace().collect();
    if parts.len() <= 1 {
        return vec![];
    }
    let mut paths = Vec::new();
    let mut i = 1;
    while i < parts.len() {
        let p = parts[i];
        if p == "-p" {
            i += 1;
            continue;
        }
        if p.starts_with('-') {
            i += 1;
            continue;
        }
        paths.push(p.to_string());
        i += 1;
    }
    paths
}

/// True when every explicit path argument is an absolute path under approved temp.
fn scratch_paths_under_temp(segment: &str) -> bool {
    let paths = non_flag_path_args(segment);
    !paths.is_empty()
        && paths.iter().all(|p| {
            let path = Path::new(p);
            path.is_absolute() && crate::tools::is_path_under_allowed_temp(path)
        })
}

/// Check a single command segment for unsafe operations.
fn check_segment(segment: &str) -> Result<(), String> {
    let trimmed = segment.trim();
    if trimmed.is_empty() {
        return Ok(());
    }

    // Note: redirect detection is done at the command level in check_command(),
    // not per-segment, because compound operators like >| span segment boundaries.

    // Extract the effective command by stripping shell prefixes and
    // environment variable assignments.
    let canonical = super::canonical_command(trimmed);
    let first_word = canonical.split_whitespace().next().unwrap_or("");

    if first_word.is_empty() {
        return Ok(());
    }

    // 'mktemp' creates a temp directory and outputs its path — always allowed.
    if first_word == "mktemp" {
        return Ok(());
    }

    // Check unconditional rejection list
    if MUTATING_COMMANDS.contains(&first_word) {
        if SCRATCH_MUTATORS.contains(&first_word) && scratch_paths_under_temp(trimmed) {
            return Ok(());
        }
        return reject(
            trimmed,
            &format!("`{first_word}` is not allowed — it modifies the workspace."),
            "use read-only alternatives to inspect files, e.g. `cat`, `head`, `tail`, `ls`, `file`, `stat`.",
        );
    }

    // Git-specific checks
    if first_word == "git" {
        return check_git_segment(trimmed);
    }

    // Cargo-specific checks
    if first_word == "cargo" {
        return check_cargo_segment(trimmed);
    }

    // Flag-dependent checks: reject commands that use mutation flags.
    // Every guarded arm returns early; the `_ => {}` fallthrough leads to the
    // trailing `Ok(())` for the allow case.
    match first_word {
        "sed" if has_flag(trimmed, "i") => {
            return reject(
                trimmed,
                "`sed -i` is not allowed — it modifies files in-place.",
                "use `sed` without `-i` to output to stdout, e.g. `sed 's/a/b/' file`.",
            );
        }
        "awk" if has_inplace(trimmed) => {
            return reject(
                trimmed,
                "`awk -i inplace` is not allowed — it modifies files in-place.",
                "use `awk` without `-i inplace` to output to stdout.",
            );
        }
        "dd" if has_dd_of(trimmed) => {
            return reject(
                trimmed,
                "`dd of=...` is not allowed — it writes to a file.",
                "use `dd` without `of=` to output to stdout.",
            );
        }
        "curl" if has_output_flag(trimmed) => {
            return reject(
                trimmed,
                "`curl` with output flags (`-o`, `--output`, `-O`, `--remote-name`) is not allowed.",
                "use `curl` without output flags to display content in stdout.",
            );
        }
        "tar" if !is_tar_list_only(trimmed) => {
            return reject(
                trimmed,
                "`tar` is only allowed with `-t`/`--list` (list) mode.",
                "use `tar -tf archive.tar` to list contents.",
            );
        }
        "base64" if has_base64_decode_output(trimmed) => {
            return reject(
                trimmed,
                "`base64 -d` with `-o` is not allowed — it writes decoded output to a file.",
                "use `base64 -d` without `-o` to output to stdout.",
            );
        }
        _ => {}
    }

    Ok(())
}

// ── Git-specific checks ──────────────────────────────────────────────────

/// Mutation flags/verbs for `git branch` (any of these makes the command mutating).
const GIT_BRANCH_MUTATIONS: &[&str] = &[
    "-d",
    "-D",
    "-m",
    "-M",
    "-c",
    "-C",
    "--delete",
    "--move",
    "--copy",
    "--edit-description",
];

/// Mutation flags for `git tag` (any of these makes the command mutating).
const GIT_TAG_MUTATIONS: &[&str] = &[
    "-d",
    "--delete",
    "-a",
    "-s",
    "-u",
    "--annotate",
    "--sign",
    "--local-user",
];

/// Mutation verbs for `git remote` (any of these makes the command mutating).
const GIT_REMOTE_MUTATIONS: &[&str] = &[
    "add",
    "remove",
    "rm",
    "rename",
    "set-url",
    "set-head",
    "set-branches",
    "update",
    "prune",
];

fn check_git_segment(segment: &str) -> Result<(), String> {
    let trimmed = segment.trim();

    // Special case: `git stash list` is safe
    if trimmed.contains("stash list") {
        return Ok(());
    }

    // Extract the git subcommand by skipping "git" and global flags
    let subcommand = extract_git_subcommand(trimmed);

    if subcommand.is_empty() || subcommand == "git" {
        return Ok(());
    }

    // git stash (without "list") is rejected
    if subcommand.starts_with("stash") && !subcommand.contains("stash list") {
        return reject(
            trimmed,
            "`git stash` is not allowed — it modifies the working tree.",
            "use `git stash list` to view stashes, or `git diff` to preview changes.",
        );
    }

    // Check if the subcommand is safe
    let mut matched_safe = "";
    for safe in GIT_SAFE_SUBCOMMANDS {
        if subcommand == *safe || subcommand.starts_with(&format!("{safe} ")) {
            matched_safe = safe;
            break;
        }
    }

    if matched_safe.is_empty() {
        return Err(format!(
            "⚠️ Read-only mode: the `git {subcommand}` subcommand is not allowed — it may mutate the repository.\n\
             Command: `{trimmed}`\n\
             Allowed git subcommands for read-only mode: status, log, diff, show, blame, branch, tag, remote, stash list,\n\
             and other inspection-only commands. Suggestion: use these for repository exploration."
        ));
    }

    // Additional mutation-flag checks for branch/tag/remote
    match matched_safe {
        "branch" => check_git_subcommand_mutation(&subcommand, "branch", GIT_BRANCH_MUTATIONS)?,
        "tag" => check_git_subcommand_mutation(&subcommand, "tag", GIT_TAG_MUTATIONS)?,
        "remote" => check_git_subcommand_mutation(&subcommand, "remote", GIT_REMOTE_MUTATIONS)?,
        _ => {}
    }

    Ok(())
}

/// For a matched git subcommand, check if the next token after the
/// subcommand name is a mutation flag/verb. Reject if it is.
///
/// `subcommand` is the pre-extracted subcommand from [`extract_git_subcommand`]
/// (e.g., `"branch -d feature"`).
fn check_git_subcommand_mutation(
    subcommand: &str,
    subcommand_name: &str,
    mutation_tokens: &[&str],
) -> Result<(), String> {
    let words: Vec<&str> = subcommand.split_whitespace().collect();
    // words[0] is the subcommand name (e.g., "branch")
    // Check the first argument for a mutation token
    if let Some(first_arg) = words.get(1)
        && mutation_tokens.contains(first_arg)
    {
        return Err(format!(
            "⚠️ Read-only mode: `git {subcommand}` is not allowed — it mutates.\n\
             Suggestion: use `git {subcommand_name}` without mutation flags to list/inspect."
        ));
    }
    // Only check the first argument — if it's safe, the command is safe
    // (best-effort: `git branch --sort=-committerdate` passes as read-only)
    Ok(())
}

/// Extract the full subcommand from a git segment.
///
/// Skips leading environment variable assignments, skips the `git` command
/// word, skips global flags and their values, and collects all remaining
/// words as the subcommand.
fn extract_git_subcommand(segment: &str) -> String {
    let words: Vec<&str> = segment.split_whitespace().collect();

    // Skip leading env assignments to find "git" (e.g., GIT_DIR=/tmp git push).
    let git_idx = words.iter().position(|w| !super::is_env_assignment(w));
    if git_idx.is_none_or(|idx| words[idx] != "git") {
        return String::new();
    }
    let git_idx = git_idx.unwrap();

    // Use shared helper to skip git global flags and other flags,
    // then take all remaining words as the subcommand verbatim.
    let remaining = &words[git_idx + 1..];
    if let Some(sub_start) = super::find_first_non_flag_index(remaining, true) {
        remaining[sub_start..].join(" ")
    } else {
        String::new()
    }
}

// ── Cargo-specific checks ─────────────────────────────────────────────────

fn check_cargo_segment(segment: &str) -> Result<(), String> {
    let trimmed = segment.trim();
    let canonical = super::canonical_command(trimmed);
    let subcommand = canonical.strip_prefix("cargo ").unwrap_or(&canonical);

    if subcommand.is_empty() || subcommand == "cargo" {
        return Ok(());
    }

    // Extract base subcommand (first word)
    let base = subcommand.split_whitespace().next().unwrap_or("");

    // Check if the subcommand is in the safe list
    let is_safe = CARGO_SAFE_SUBCOMMANDS.contains(&base);

    if !is_safe {
        return Err(format!(
            "⚠️ Read-only mode: `cargo {base}` is not in the allowed cargo subcommands list.\n\
             Command: `{trimmed}`\n\
             Allowed cargo subcommands: {}\n\
             Suggestion: use `cargo check`, `cargo test`, `cargo clippy`, `cargo doc`, etc.",
            CARGO_SAFE_SUBCOMMANDS.join(", ")
        ));
    }

    // cargo clippy --fix rejection (only when --fix appears BEFORE --)
    if base == "clippy" && has_clippy_fix(trimmed) {
        return Err(format!(
            "⚠️ Read-only mode: `cargo clippy --fix` is not allowed — it auto-applies fixes.\n\
             Command: `{trimmed}`\n\
             Suggestion: use `cargo clippy` without `--fix` to see warnings only,\n\
             or use `cargo clippy -- --fix` to pass `--fix` as a lint name (not auto-fix)."
        ));
    }

    // cargo fmt without --check rejection
    if base == "fmt" && !has_cargo_fmt_check(trimmed) {
        return reject(
            trimmed,
            "`cargo fmt` without `--check` is not allowed — it reformats files.",
            "use `cargo fmt --check` to verify formatting without modifying files.",
        );
    }

    Ok(())
}

// ── Flag detection helpers ────────────────────────────────────────────────

/// Check if the command has the given short flag (e.g., `-i`, including `-i.bak` variant).
fn has_flag(command: &str, flag: &str) -> bool {
    let parts: Vec<&str> = command.split_whitespace().collect();
    let dash_flag = format!("-{flag}");
    for part in &parts {
        if *part == dash_flag || part.starts_with(&format!("-{flag}.")) {
            return true;
        }
    }
    false
}

/// Check if `awk -i inplace` is present.
fn has_inplace(command: &str) -> bool {
    let parts: Vec<&str> = command.split_whitespace().collect();
    for i in 0..parts.len().saturating_sub(1) {
        if parts[i] == "-i" && parts[i + 1] == "inplace" {
            return true;
        }
    }
    false
}

/// Check if `dd of=...` is present.
fn has_dd_of(command: &str) -> bool {
    command.split_whitespace().any(|p| p.starts_with("of="))
}

/// Check if curl/wget has output flags.
fn has_output_flag(command: &str) -> bool {
    let parts: Vec<&str> = command.split_whitespace().collect();
    parts
        .iter()
        .any(|p| *p == "-o" || *p == "--output" || *p == "-O" || *p == "--remote-name")
}

/// Check if tar is using only `-t`/`--list` (list) mode. Handles combined flags.
fn is_tar_list_only(command: &str) -> bool {
    let parts: Vec<&str> = command.split_whitespace().collect();
    // Find the operation flag/option
    for part in &parts {
        // --list is always safe
        if *part == "--list" {
            return true;
        }
        if part.starts_with('-') && !part.starts_with("--") {
            // Skip non-operation flags
            if *part == "-v" || *part == "-f" || *part == "-z" || *part == "-j" || *part == "-J" {
                continue;
            }
            // Check if this contains only 't' (and maybe v/f/z/j/J) as operation flags
            let ops: String = part
                .chars()
                .skip(1) // skip leading '-'
                .filter(|c| !['v', 'f', 'z', 'j', 'J'].contains(c))
                .collect();
            if !ops.is_empty() {
                return ops == "t";
            }
        }
    }
    // No operation flag found — reject (conservative)
    false
}

/// Check if `base64` has both decode flag (`-d`/`--decode`) and output flag
/// (`-o`/`--output`), which would write decoded data to a file.
fn has_base64_decode_output(command: &str) -> bool {
    let parts: Vec<&str> = command.split_whitespace().collect();
    let has_d = parts.iter().any(|p| *p == "-d" || *p == "--decode");
    let has_o = parts.iter().any(|p| *p == "-o" || *p == "--output");
    has_d && has_o
}

/// Check if `cargo clippy` has `--fix` before `--`.
fn has_clippy_fix(command: &str) -> bool {
    let parts: Vec<&str> = command.split_whitespace().collect();
    let dashdash_pos = parts.iter().position(|p| *p == "--");
    for (i, part) in parts.iter().enumerate() {
        if *part == "--fix" {
            // If --fix appears before --, it's the auto-fix flag
            // If --fix appears after --, it's a lint name
            if let Some(dd_pos) = dashdash_pos
                && i > dd_pos
            {
                return false; // after -- = lint name
            }
            return true; // before -- (or no --) = auto-fix
        }
    }
    false
}

/// Check if `cargo fmt` has `--check` anywhere in args.
fn has_cargo_fmt_check(command: &str) -> bool {
    command.split_whitespace().any(|p| p == "--check")
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(cmd: &str) {
        assert!(
            check_command(cmd).is_ok(),
            "expected ALLOW but got REJECT for: `{cmd}`"
        );
    }

    fn assert_rejected(cmd: &str) {
        assert!(
            check_command(cmd).is_err(),
            "expected REJECT but got ALLOW for: `{cmd}`"
        );
    }

    // ── Empty / whitespace ──────────────────────────────────────────

    #[test]
    fn empty_command() {
        ok("");
    }

    #[test]
    fn whitespace_only() {
        ok("   ");
    }

    #[test]
    fn unknown_command_allowed() {
        ok("some_obscure_tool --flag");
    }

    // ── Git allowlist ──────────────────────────────────────────────

    #[test]
    fn git_status() {
        ok("git status");
    }

    #[test]
    fn git_log_oneline() {
        ok("git log --oneline");
    }

    #[test]
    fn git_diff() {
        ok("git diff HEAD~1");
    }

    #[test]
    fn git_commit_rejected() {
        assert_rejected("git commit -m test");
    }

    #[test]
    fn git_push_rejected() {
        assert_rejected("git push");
    }

    #[test]
    fn git_stash_rejected() {
        assert_rejected("git stash");
    }

    #[test]
    fn git_stash_list_allowed() {
        ok("git stash list");
    }

    #[test]
    fn git_branch_allowed() {
        ok("git branch");
    }

    #[test]
    fn git_tag_allowed() {
        ok("git tag");
    }

    #[test]
    fn git_remote_allowed() {
        ok("git remote");
    }

    #[test]
    fn git_blame_allowed() {
        ok("git blame src/main.rs");
    }

    #[test]
    fn git_cat_file_allowed() {
        ok("git cat-file -p HEAD");
    }

    #[test]
    fn git_worktree_list_allowed() {
        ok("git worktree list");
    }

    #[test]
    fn git_config_list_allowed() {
        ok("git config --list");
    }

    #[test]
    fn git_config_get_allowed() {
        ok("git config --get user.name");
    }

    #[test]
    fn git_merge_rejected() {
        assert_rejected("git merge feature");
    }

    #[test]
    fn git_rebase_rejected() {
        assert_rejected("git rebase main");
    }

    // ── Cargo allowlist ────────────────────────────────────────────

    #[test]
    fn cargo_check() {
        ok("cargo check");
    }

    #[test]
    fn cargo_test() {
        ok("cargo test");
    }

    #[test]
    fn cargo_clippy() {
        ok("cargo clippy");
    }

    #[test]
    fn cargo_clippy_fix_rejected() {
        assert_rejected("cargo clippy --fix");
    }

    #[test]
    fn cargo_clippy_fix_after_dd_allowed() {
        ok("cargo clippy -- --fix");
    }

    #[test]
    fn cargo_fmt_rejected() {
        assert_rejected("cargo fmt");
    }

    #[test]
    fn cargo_fmt_check_allowed() {
        ok("cargo fmt --check");
    }

    #[test]
    fn cargo_fmt_dd_check_allowed() {
        ok("cargo fmt -- --check");
    }

    #[test]
    fn cargo_fix_rejected() {
        assert_rejected("cargo fix");
    }

    #[test]
    fn cargo_doc_allowed() {
        ok("cargo doc");
    }

    #[test]
    fn cargo_update_allowed() {
        ok("cargo update");
    }

    #[test]
    fn cargo_clean_allowed() {
        ok("cargo clean");
    }

    // ── Unconditional rejections ──────────────────────────────────

    /// Tests that ALL entries in the production [`MUTATING_COMMANDS`] constant
    /// are rejected. Iterates the constant directly to prevent coverage drift
    /// when entries are added or removed.
    #[test]
    fn all_mutating_commands_rejected() {
        for cmd in MUTATING_COMMANDS {
            assert_rejected(&format!("{cmd} arg"));
        }
    }

    /// Tests that all git branch mutation flags are rejected via
    /// [`check_git_subcommand_mutation`].
    #[test]
    fn git_branch_mutation_flags_rejected() {
        for flag in GIT_BRANCH_MUTATIONS {
            assert_rejected(&format!("git branch {flag} feature"));
        }
    }

    /// Tests that all git tag mutation flags are rejected via
    /// [`check_git_subcommand_mutation`].
    #[test]
    fn git_tag_mutation_flags_rejected() {
        for flag in GIT_TAG_MUTATIONS {
            assert_rejected(&format!("git tag {flag} v1.0"));
        }
    }

    /// Tests that all git remote mutation verbs are rejected via
    /// [`check_git_subcommand_mutation`].
    #[test]
    fn git_remote_mutation_verbs_rejected() {
        for verb in GIT_REMOTE_MUTATIONS {
            assert_rejected(&format!("git remote {verb} origin"));
        }
    }

    // ── Flag-dependent tests ──────────────────────────────────────

    #[test]
    fn sed_stdout_allowed() {
        ok("sed 's/a/b/' file");
    }

    #[test]
    fn sed_inplace_rejected() {
        assert_rejected("sed -i 's/a/b/' file");
    }

    #[test]
    fn sed_inplace_bak_rejected() {
        assert_rejected("sed -i.bak 's/a/b/' file");
    }

    #[test]
    fn awk_stdout_allowed() {
        ok("awk '{print $1}' file");
    }

    #[test]
    fn awk_inplace_rejected() {
        assert_rejected("awk -i inplace '{print $1}' file");
    }

    #[test]
    fn dd_stdout_allowed() {
        ok("dd if=/dev/zero bs=1 count=10");
    }

    #[test]
    fn dd_of_rejected() {
        assert_rejected("dd if=/dev/zero of=file bs=1 count=10");
    }

    #[test]
    fn curl_allowed() {
        ok("curl https://example.com");
    }

    #[test]
    fn curl_output_rejected() {
        assert_rejected("curl -o file https://example.com");
    }

    #[test]
    fn curl_remote_name_rejected() {
        assert_rejected("curl -O https://example.com/file");
    }

    #[test]
    fn tar_list_allowed() {
        ok("tar -tf archive.tar.gz");
    }

    #[test]
    fn tar_extract_rejected() {
        assert_rejected("tar -xzf archive.tar.gz");
    }

    #[test]
    fn tar_create_rejected() {
        assert_rejected("tar -czf archive.tar.gz dir/");
    }

    #[test]
    fn base64_decode_stdout_allowed() {
        ok("base64 -d file.txt");
    }

    #[test]
    fn base64_decode_with_output_rejected() {
        assert_rejected("base64 -d -o out.bin file.txt");
    }

    #[test]
    fn base64_decode_long_output_rejected() {
        assert_rejected("base64 --decode --output out.bin file.txt");
    }

    // ── Chained commands ───────────────────────────────────────────

    #[test]
    fn chained_all_safe() {
        ok("cargo check && cargo test");
    }

    #[test]
    fn chained_second_mutates() {
        assert_rejected("cargo check && rm file");
    }

    #[test]
    fn chained_second_mutates_fmt() {
        assert_rejected("git status && cargo fmt");
    }

    #[test]
    fn piped_all_safe() {
        ok("git log --oneline | head -20");
    }

    #[test]
    fn semicolon_second_mutates() {
        assert_rejected("cargo check; rm file");
    }

    // ── Redirect tests ─────────────────────────────────────────────

    #[test]
    fn redirect_to_workspace_rejected() {
        assert_rejected("echo hello > file.txt");
    }

    #[test]
    fn redirect_to_devnull_allowed() {
        ok("echo hello > /dev/null");
    }

    #[test]
    fn redirect_to_tmp_allowed() {
        ok("echo hello > /tmp/output.txt");
    }

    #[test]
    fn stderr_to_stdout_allowed() {
        ok("cmd 2>&1");
    }

    #[test]
    fn redirect_in_quotes_allowed() {
        ok("echo \"hello > world\"");
    }

    #[test]
    fn append_to_tmp_allowed() {
        ok("echo hello >> /tmp/log");
    }

    #[test]
    fn noclobber_to_tmp_allowed() {
        ok("echo hello >| /tmp/force");
    }

    #[test]
    fn stdout_stderr_to_devnull() {
        ok("cargo build > /dev/null 2>&1");
    }

    // ── mktemp (temp dir, allowed) ────────────────────────────────

    #[test]
    fn mktemp_allowed() {
        ok("mktemp");
    }

    #[test]
    fn mktemp_with_template_allowed() {
        ok("mktemp -t mahbot.XXXXXX");
    }

    // ── Prefix stripping (P0) ──────────────────────────────────────

    #[test]
    fn sudo_rm_rejected() {
        assert_rejected("sudo rm file");
    }

    #[test]
    fn sudo_flag_rm_rejected() {
        assert_rejected("sudo -E rm file");
    }

    #[test]
    fn env_rm_rejected() {
        assert_rejected("env rm file");
    }

    #[test]
    fn exec_rm_rejected() {
        assert_rejected("exec rm file");
    }

    #[test]
    fn nohup_rm_rejected() {
        assert_rejected("nohup rm file");
    }

    #[test]
    fn command_rm_rejected() {
        assert_rejected("command rm file");
    }

    #[test]
    fn eval_rm_rejected() {
        assert_rejected("eval rm file");
    }

    #[test]
    fn sudo_git_status_allowed() {
        ok("sudo git status");
    }

    #[test]
    fn sudo_cargo_check_allowed() {
        ok("sudo cargo check");
    }

    #[test]
    fn pure_prefix_allowed() {
        ok("cd"); // no command after prefix — harmless
    }

    #[test]
    fn cd_some_dir_allowed() {
        ok("cd .."); // builtin, no real command extracted
    }

    // ── VAR=val stripping (P0) ─────────────────────────────────────

    #[test]
    fn env_var_rm_rejected() {
        assert_rejected("FOO=bar rm file");
    }

    #[test]
    fn env_var_sudo_rm_rejected() {
        assert_rejected("VAR=val sudo rm -rf /");
    }

    #[test]
    fn env_var_git_status_allowed() {
        ok("GIT_DIR=/tmp git status");
    }

    // ── Script interpreters: read-only usage (not blocked) ─────────

    #[test]
    fn python3_version_allowed() {
        ok("python3 --version");
    }

    #[test]
    fn python3_print_allowed() {
        ok("python3 -c \"print('hello')\"");
    }

    #[test]
    fn node_eval_allowed() {
        ok("node -e \"console.log('hi')\"");
    }

    #[test]
    fn bash_echo_allowed() {
        ok("bash -c \"echo hello\"");
    }

    // ── Container tools: read-only usage (not blocked) ──────────────

    #[test]
    fn docker_ps_allowed() {
        ok("docker ps");
    }

    #[test]
    fn kubectl_get_allowed() {
        ok("kubectl get pods");
    }

    // ── Git branch/tag/remote mutation flags (P3) ───────────────────

    #[test]
    fn git_branch_list_allowed() {
        ok("git branch");
    }

    #[test]
    fn git_branch_sort_allowed() {
        ok("git branch --sort=-committerdate");
    }

    #[test]
    fn git_tag_list_allowed() {
        ok("git tag");
    }

    #[test]
    fn git_remote_list_allowed() {
        ok("git remote");
    }

    #[test]
    fn git_remote_verbose_allowed() {
        ok("git remote -v");
    }

    // ── New safe subcommands ───────────────────────────────────────

    #[test]
    fn git_reflog_allowed() {
        ok("git reflog");
    }

    #[test]
    fn git_range_diff_allowed() {
        ok("git range-diff HEAD~3 HEAD~1 HEAD");
    }

    #[test]
    fn cargo_version_allowed() {
        ok("cargo version");
    }

    #[test]
    fn cargo_help_allowed() {
        ok("cargo help build");
    }

    #[test]
    fn cargo_bench_allowed() {
        ok("cargo bench");
    }

    // ── tar --list (P9) ────────────────────────────────────────────

    #[test]
    fn tar_long_list_allowed() {
        ok("tar --list -f archive.tar.gz");
    }

    // ── Redirect /var/tmp (P9) ─────────────────────────────────────

    #[test]
    fn redirect_to_var_tmp_allowed() {
        ok("echo hello > /var/tmp/output.txt");
    }

    #[test]
    fn append_to_var_tmp_allowed() {
        ok("echo hello >> /var/tmp/log");
    }

    // ── Redirect operators added for has_disallowed_redirect refactor ──
    //
    // These test previously-uncovered redirect operators and edge cases.
    // `has_disallowed_redirect` was refactored to use a shared quote-tracking
    // helper (`check_outside_quotes`) and a char iterator (fixing a pre-existing
    // multi-byte UTF-8 bug where `bytes[i] as char` produced garbage for
    // non-ASCII).

    #[test]
    fn redirect_bare_gt_rejected() {
        // Bare > redirect to relative path = disallowed
        assert_rejected("cmd > output.txt");
    }

    #[test]
    fn redirect_fd_merge_stderr_to_stdout_allowed() {
        // 1>&2 is a pure fd merge — same as 2>&1 but reversed
        ok("cmd 1>&2");
    }

    #[test]
    fn redirect_2gt_to_tmp_allowed() {
        ok("cmd 2> /tmp/errors.log");
    }

    #[test]
    fn redirect_2gt_to_workspace_rejected() {
        assert_rejected("cmd 2> errors.log");
    }

    #[test]
    fn redirect_gt_ampersand_rejected() {
        // `>&2` redirects stdout to stderr (same fd, but `>&` with
        // non-temp target = not explicitly allowed)
        assert_rejected("cmd >&2");
    }

    #[test]
    fn backslash_escaped_redirect_not_detected() {
        // echo \> /tmp/file — the > is backslash-escaped, not a redirect
        ok("echo \\> /tmp/file");
    }

    #[test]
    fn backslash_escaped_redirect_not_detected_no_target() {
        // echo \> — bare but escaped, not a redirect
        ok("echo \\>");
    }

    #[test]
    fn multiple_consecutive_backslash_escapes() {
        // \\\\> — double backslash escapes \, then \ escapes >, so > is not a redirect
        ok("echo \\\\\\> file");
    }

    #[test]
    fn unclosed_double_quote_hides_redirect() {
        // redirect operator inside unclosed double quotes = not detected
        ok("echo \"> /tmp/foo");
    }

    #[test]
    fn unclosed_single_quote_hides_redirect() {
        // redirect operator inside unclosed single quotes = not detected
        ok("echo '> /tmp/foo");
    }

    // ── extract_git_subcommand unit tests ──────────────────────────

    #[test]
    fn extract_git_subcommand_basic() {
        assert_eq!(extract_git_subcommand("git status"), "status");
    }

    #[test]
    fn extract_git_subcommand_with_global_flag() {
        assert_eq!(extract_git_subcommand("git -C /repo diff"), "diff");
    }

    #[test]
    fn extract_git_subcommand_with_config() {
        assert_eq!(extract_git_subcommand("git -c user.name=me log"), "log");
    }

    #[test]
    fn extract_git_subcommand_with_git_dir() {
        assert_eq!(
            extract_git_subcommand("git --git-dir /repo status"),
            "status"
        );
    }

    #[test]
    fn extract_git_subcommand_env_assignment() {
        assert_eq!(extract_git_subcommand("GIT_DIR=/tmp git status"), "status");
    }

    #[test]
    fn extract_git_subcommand_no_git() {
        assert_eq!(extract_git_subcommand("cargo build"), "");
    }

    #[test]
    fn extract_git_subcommand_git_only() {
        assert_eq!(extract_git_subcommand("git"), "");
    }

    #[test]
    fn extract_git_subcommand_full_subcommand() {
        assert_eq!(
            extract_git_subcommand("git branch -d feature"),
            "branch -d feature"
        );
    }

    #[test]
    fn extract_git_subcommand_with_double_dash() {
        assert_eq!(extract_git_subcommand("git -- diff"), "diff");
    }

    #[test]
    fn extract_git_subcommand_stash_list() {
        assert_eq!(extract_git_subcommand("git stash list"), "stash list");
    }

    #[test]
    fn extract_git_subcommand_multiple_env() {
        assert_eq!(
            extract_git_subcommand("CC=gcc CXX=g++ git status"),
            "status"
        );
    }

    #[test]
    fn extract_git_subcommand_multiple_flags() {
        assert_eq!(
            extract_git_subcommand("git -C /repo --git-dir /other status"),
            "status"
        );
    }

    #[test]
    fn extract_git_subcommand_shell_prefix_not_skipped() {
        // Shell prefixes like `sudo` are NOT skipped — only env assignments.
        // `sudo` is not "git", so this returns empty (read-only validation
        // will reject the command as unknown).
        assert_eq!(extract_git_subcommand("sudo git status"), "");
    }

    #[test]
    fn extract_git_subcommand_flag_with_multiple_args() {
        assert_eq!(
            extract_git_subcommand("git branch --merged master"),
            "branch --merged master"
        );
    }

    #[test]
    fn heredoc_to_tmp_with_rust_body_allowed() {
        ok("cat > /tmp/test_match.rs << 'EOF'\nfn test() { match x { \"a\" => 1, _ => 0 } }\nEOF");
    }

    #[test]
    fn redirect_to_private_tmp_allowed() {
        ok("echo hello > /private/tmp/mahbot_test_out.txt");
    }

    #[test]
    fn tee_under_tmp_allowed() {
        ok("tee /tmp/scratch.log");
    }

    #[test]
    fn touch_under_tmp_allowed() {
        ok("touch /tmp/scratch.txt");
    }

    #[test]
    fn mkdir_p_under_tmp_allowed() {
        ok("mkdir -p /tmp/scratch_dir");
    }

    #[test]
    fn tee_workspace_rejected() {
        assert_rejected("tee output.log");
    }

    #[test]
    fn rm_under_tmp_still_rejected() {
        assert_rejected("rm /tmp/scratch.txt");
    }
}
