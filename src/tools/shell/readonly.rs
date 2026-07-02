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
    "stash list",
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
///
/// # Security invariant
///
/// This function MUST distinguish `<<` outside quotes (real heredoc) from `<<`
/// inside quotes (literal text).  Failure to do so creates a false-negative
/// security bypass: a quoted `<<` causes everything after it (including real
/// redirect operators) to be removed from the scan string, making
/// [`has_disallowed_redirect`] miss the redirect.
///
/// # Known limitation (pre-existing, not addressed here)
///
/// - Heredoc bodies that contain the delimiter within quotes are not detected
///   (the body-skipping loop checks for literal delimiter matches).  In a real
///   shell, a quoted delimiter in the body does NOT terminate the heredoc.
///   This can produce false negatives (allowing a dangerous redirect inside a
///   heredoc body whose delimiter appears inside quotes earlier in the body),
///   but such multi-line engineered inputs are unlikely in practice.
fn strip_heredoc_bodies(command: &str) -> String {
    let mut out = String::new();
    let mut i = 0;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let chars: Vec<(usize, char)> = command.char_indices().collect();

    while i < chars.len() {
        // ── Escape tracking ────────────────────────────────────────
        // Must come before quote state tracking so escaped quotes
        // (`\'`, `\"`) don't toggle in_single/in_double.  The
        // `!in_single` guard means backslash is treated as escape both
        // outside quotes and inside double quotes (inside double quotes,
        // `\` should only escape `\`, `$`, `` ` ``, `"`, and newline,
        // but treating any backslash as escape is a safe over-approximation:
        // the escaped char is preserved in output and skipped for quote
        // state / heredoc detection; at worst it causes a false negative
        // (missed redirect) which is acceptable for a best-effort layer).
        // Inside single quotes, backslash is always literal.
        //
        // When a character is escaped, we still push it to the output
        // (to preserve the command string for redirect scanning), but we
        // skip quote-state tracking and heredoc detection for it.
        // This mirrors the philosophy of [`has_disallowed_redirect`]'s
        // escape handling: over-escaping is safe (false negative = allow,
        // which is acceptable for this best-effort safety layer).
        if escaped {
            escaped = false;
            out.push(chars[i].1);
            i += 1;
            continue;
        }
        if chars[i].1 == '\\' && !in_single {
            escaped = true;
            out.push(chars[i].1);
            i += 1;
            continue;
        }

        // ── Quote state tracking ───────────────────────────────────
        // [`check_outside_quotes`] returns `false` both for quote
        // characters (`'`, `"`) and for characters inside quotes.
        // When inside quotes, we push the character to output and
        // skip heredoc detection — `<<` inside quotes is literal text,
        // not a heredoc start.
        if !super::check_outside_quotes(chars[i].1, &mut in_single, &mut in_double) {
            out.push(chars[i].1);
            i += 1;
            continue;
        }

        // ── Heredoc detection (only outside quotes) ────────────────
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
            if crate::tools::path::is_path_under_allowed_temp(target_path) {
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

/// A flag-dependent command check: if the command's first word matches `verb`
/// and the `predicate` returns true, the command is rejected with the given message.
struct FlagCheck {
    verb: &'static str,
    predicate: fn(&str) -> bool,
    rejection: &'static str,
    suggestion: &'static str,
}

/// Flag-dependent checks: reject commands that use mutation flags.
/// Each entry tests a specific verb + predicate combination.
const FLAG_CHECKS: &[FlagCheck] = &[
    FlagCheck {
        verb: "sed",
        predicate: sed_has_flag_i,
        rejection: "`sed -i` is not allowed — it modifies files in-place.",
        suggestion: "use `sed` without `-i` to output to stdout, e.g. `sed 's/a/b/' file`.",
    },
    FlagCheck {
        verb: "awk",
        predicate: has_inplace,
        rejection: "`awk -i inplace` is not allowed — it modifies files in-place.",
        suggestion: "use `awk` without `-i inplace` to output to stdout.",
    },
    FlagCheck {
        verb: "dd",
        predicate: has_dd_of,
        rejection: "`dd of=...` is not allowed — it writes to a file.",
        suggestion: "use `dd` without `of=` to output to stdout.",
    },
    FlagCheck {
        verb: "curl",
        predicate: has_curl_output_flag,
        rejection: "`curl` with output flags (`-o`, `--output`, `-O`, `--remote-name`) is not allowed.",
        suggestion: "use `curl` without output flags to display content in stdout.",
    },
    FlagCheck {
        verb: "tar",
        predicate: is_not_tar_list_only,
        rejection: "`tar` is only allowed with `-t`/`--list` (list) mode.",
        suggestion: "use `tar -tf archive.tar` to list contents.",
    },
    FlagCheck {
        verb: "base64",
        predicate: has_base64_decode_output,
        rejection: "`base64 -d` with `-o` is not allowed — it writes decoded output to a file.",
        suggestion: "use `base64 -d` without `-o` to output to stdout.",
    },
];

/// Collect all non-flag, non-redirect, non-heredoc path-like arguments from a
/// command segment, scanning the **original** whitespace-split tokens.
///
/// This replaces the previous implementation that used [`canonical_command`],
/// which truncated to the first non-flag argument only, meaning multiple
/// path arguments (e.g. `tee /tmp/a /etc/passwd`) had only the first one
/// validated — a security bypass (see mahbot-396).
///
/// The function skips:
/// - Shell flags (tokens starting with `-`)
/// - Standalone redirect operators that expect a target word (the next token
///   is also skipped): symbolic forms `>`, `>&`, `>>`, `>|`, `<`, `<&`, `<>`;
///   digit-prefixed forms `{digit}>`, `{digit}<` (e.g. `2>`, `10>`, `3<`);
///   bash extensions `&>`, `&>>`
/// - Self-contained redirect operators (no separate target): `2>&1`, `1>&2`
/// - Combined redirect tokens (operator merged with target, no separate word
///   to skip): e.g. `>/dev/null`, `2>/dev/null`, `</dev/null`, `<<`/`<<-` heredocs,
///   `<&2`, `<>/tmp/file`, `&>/dev/null`, `&>>file`, `{digit}<<EOF`
/// - Heredoc operators (`<<`, `<<-`, `{digit}<<`) and everything after them
///   (delimiter, body, terminating delimiter).  **Limitation:** path arguments
///   that appear after the heredoc terminator are not validated (e.g.
///   `tee /tmp/a << 'EOF'\nbody\nEOF /etc/passwd`).  This is consistent with
///   the best-effort security model documented in [`check_command`].
///
///   The same conservative skip-everything-after-heredoc-operator applies to
///   fd-prefixed heredocs (`3<<EOF`, `1<<-EOF`).
fn non_flag_path_args(segment: &str) -> Vec<String> {
    let words: Vec<&str> = segment.split_whitespace().collect();
    let Some(cmd_idx) = super::find_first_command_word_index(&words) else {
        return vec![];
    };

    let mut args = Vec::new();
    let mut skip_redirect_target = false;
    let mut in_heredoc_body = false;

    for w in &words[cmd_idx + 1..] {
        if in_heredoc_body {
            continue;
        }

        if skip_redirect_target {
            skip_redirect_target = false;
            continue;
        }

        if w.starts_with('-') {
            continue;
        }

        // ── Heredoc detection ───────────────────────────────────────
        if w.starts_with("<<") {
            in_heredoc_body = true;
            continue;
        }
        // Heredoc with fd prefix (e.g. 3<<EOF, 1<<-EOF)
        if w.len() > 2 && w.as_bytes()[0].is_ascii_digit() && w.contains("<<") {
            in_heredoc_body = true;
            continue;
        }

        // ── Redirect detection ──────────────────────────────────────
        // Standalone output redirect operators: symbolic or digit-prefixed
        if matches!(*w, ">" | ">&" | ">>" | ">|") || is_digit_suffix_redirect(w, b'>') {
            skip_redirect_target = true;
            continue;
        }
        // Standalone input redirect operators: symbolic or digit-prefixed
        if matches!(*w, "<" | "<&" | "<>") || is_digit_suffix_redirect(w, b'<') {
            skip_redirect_target = true;
            continue;
        }
        // Self-contained fd-merge redirects — no separate target
        if matches!(*w, "2>&1" | "1>&2") {
            continue;
        }
        // Combined redirect tokens: operator merged with target
        // (e.g. >/dev/null, >>file, </dev/null, <&2, <>file)
        if w.starts_with('>') || w.starts_with('<') {
            continue;
        }
        // Combined fd+redirect like 2>/dev/null, 1>/tmp/out, 3</dev/null
        if w.len() > 1 && w.as_bytes()[0].is_ascii_digit() && (w.contains('>') || w.contains('<')) {
            continue;
        }
        // Bash &> standalone redirect (space-separated target expected)
        if matches!(*w, "&>" | "&>>") {
            skip_redirect_target = true;
            continue;
        }
        // Bash &> combined stdout+stderr redirect (e.g. &>/dev/null, &>>file)
        if w.contains("&>") {
            continue;
        }

        args.push(w.to_string());
    }

    args
}

/// True when `w` is a standalone redirect token consisting of one or more
/// digits followed by a single `>` or `<` operator character, with no other
/// content (e.g. `2>`, `10>`, `3<`).  Combined forms like `2>/dev/null` or
/// `2>&1` do not match because they have non-digit characters before the
/// trailing operator byte (or the trailing byte isn't a bare operator).
fn is_digit_suffix_redirect(w: &str, op: u8) -> bool {
    let bytes = w.as_bytes();
    if bytes.len() < 2 || !bytes[0].is_ascii_digit() || bytes[bytes.len() - 1] != op {
        return false;
    }
    // All bytes except the last must be decimal digits
    bytes[..bytes.len() - 1].iter().all(u8::is_ascii_digit)
}

/// True when every explicit path argument is an absolute path under allowed temp.
fn scratch_paths_under_temp(segment: &str) -> bool {
    let paths = non_flag_path_args(segment);
    !paths.is_empty()
        && paths.iter().all(|p| {
            let path = Path::new(p);
            path.is_absolute() && crate::tools::path::is_path_under_allowed_temp(path)
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
    let first_word = super::first_command_word(trimmed);

    if first_word.is_empty() {
        return Ok(());
    }

    // 'mktemp' creates a temp directory and outputs its path — always allowed.
    if first_word == "mktemp" {
        return Ok(());
    }

    // Check scratch mutators first (tee, touch, mkdir): allowed if all explicit
    // path arguments are under an allowed temp directory.
    if SCRATCH_MUTATORS.contains(&first_word) && scratch_paths_under_temp(trimmed) {
        return Ok(());
    }

    // Check unconditional rejection list
    if MUTATING_COMMANDS.contains(&first_word) {
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
    // Iterates the FLAG_CHECKS table; the first matching entry returns early,
    // otherwise falls through to `Ok(())` for the allow case.
    for check in FLAG_CHECKS {
        if first_word == check.verb && (check.predicate)(trimmed) {
            return reject(trimmed, check.rejection, check.suggestion);
        }
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

    // Skip shell prefixes, env assignments, and flags to find "git"
    // (e.g., GIT_DIR=/tmp git push, sudo git push, env git push).
    let git_idx = super::find_first_command_word_index(&words);
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

    let base = subcommand.split_whitespace().next().unwrap_or("");

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
    let dash_flag = format!("-{flag}");
    let dash_flag_dot = format!("-{flag}.");
    command
        .split_whitespace()
        .any(|part| part == dash_flag || part.starts_with(&dash_flag_dot))
}

/// Check if the command has any of the given exact-match flags.
fn has_any_flag(command: &str, flags: &[&str]) -> bool {
    command.split_whitespace().any(|part| flags.contains(&part))
}

/// Check if a `sed` command has the `-i` flag (in-place edit).
fn sed_has_flag_i(command: &str) -> bool {
    has_flag(command, "i")
}

/// Check if `awk -i inplace` is present.
fn has_inplace(command: &str) -> bool {
    let parts: Vec<&str> = command.split_whitespace().collect();
    parts.windows(2).any(|w| w[0] == "-i" && w[1] == "inplace")
}

/// Check if `dd of=...` is present.
fn has_dd_of(command: &str) -> bool {
    command.split_whitespace().any(|p| p.starts_with("of="))
}

/// Check if curl has output flags.
fn has_curl_output_flag(command: &str) -> bool {
    has_any_flag(command, &["-o", "--output", "-O", "--remote-name"])
}

/// Characters that are non-operation tar flags (format/output modifiers).
/// These can appear alongside the operation flag in combined forms (e.g.
/// `-tvf` combines `t` (list) with `v` (verbose) and `f` (file)).
const TAR_SAFE_CHARS: &[char] = &['v', 'f', 'z', 'j', 'J'];

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
            if part.len() == 2 && TAR_SAFE_CHARS.contains(&part.chars().nth(1).unwrap()) {
                continue;
            }
            // Check if this contains only 't' (and maybe v/f/z/j/J) as operation flags
            let ops: String = part
                .chars()
                .skip(1) // skip leading '-'
                .filter(|c| !TAR_SAFE_CHARS.contains(c))
                .collect();
            if !ops.is_empty() {
                return ops == "t";
            }
        }
    }
    // No operation flag found — reject (conservative)
    false
}

/// Check if `tar` is NOT in list-only mode (i.e., will extract/create).
fn is_not_tar_list_only(command: &str) -> bool {
    !is_tar_list_only(command)
}

/// Check if `base64` has both decode flag (`-d`/`--decode`) and output flag
/// (`-o`/`--output`), which would write decoded data to a file.
fn has_base64_decode_output(command: &str) -> bool {
    has_any_flag(command, &["-d", "--decode"]) && has_any_flag(command, &["-o", "--output"])
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
    use crate::tools::shell::{NON_DELEGATING_PREFIXES, SHELL_PREFIXES};

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

    /// Assert each case in a table-driven test.
    fn run_cases(cases: &[(&str, bool)]) {
        for &(command, allowed) in cases {
            if allowed {
                ok(command);
            } else {
                assert_rejected(command);
            }
        }
    }

    /// Assert all items in `items` are rejected when formatted with `template`.
    fn assert_all_rejected(items: &[&str], template: impl Fn(&str) -> String) {
        for &item in items {
            assert_rejected(&template(item));
        }
    }

    /// Assert all items in `items` are allowed when formatted with `template`.
    fn assert_all_allowed(items: &[&str], template: impl Fn(&str) -> String) {
        for &item in items {
            ok(&template(item));
        }
    }

    // ── Empty / whitespace ──────────────────────────────────────────

    #[test]
    fn empty_whitespace_and_unknown() {
        let cases = [
            ("", true),
            ("   ", true),
            ("some_obscure_tool --flag", true),
        ];

        run_cases(&cases);
    }

    // ── Git allowlist ──────────────────────────────────────────────

    /// Tests that ALL entries in the production [`GIT_SAFE_SUBCOMMANDS`] constant
    /// are accepted. Iterates the constant directly to prevent coverage drift
    /// when entries are added or removed.
    #[test]
    fn all_git_safe_subcommands_allowed() {
        assert_all_allowed(GIT_SAFE_SUBCOMMANDS, |subcmd| format!("git {subcmd}"));
    }

    #[test]
    fn git_individual_commands() {
        let cases = [
            ("git commit -m test", false),
            ("git push", false),
            ("git stash", false),
            ("git stash list", true),
            ("git merge feature", false),
            ("git rebase main", false),
        ];

        run_cases(&cases);
    }

    // ── Git --bare flag (regression: was skipped as a git global flag) ─

    #[test]
    fn git_bare_flag() {
        let cases = [
            ("git --bare status", true),
            ("git --bare log --oneline", true),
            ("git --bare diff", true),
            ("git --bare push", false),
            ("git --bare commit -m test", false),
            ("git --bare reset --hard", false),
        ];

        run_cases(&cases);
    }

    // ── Cargo allowlist ────────────────────────────────────────────

    /// Tests that ALL entries in the production [`CARGO_SAFE_SUBCOMMANDS`] constant
    /// (except `"fmt"`, which requires `--check`) are accepted.
    /// Iterates the constant directly to prevent coverage drift
    /// when entries are added or removed.
    #[test]
    fn all_cargo_safe_subcommands_allowed() {
        for subcmd in CARGO_SAFE_SUBCOMMANDS {
            if *subcmd == "fmt" {
                continue; // requires --check flag — tested via cargo_individual_commands
            }
            ok(&format!("cargo {subcmd}"));
        }
    }

    #[test]
    fn cargo_individual_commands() {
        let cases = [
            ("cargo clippy --fix", false),
            ("cargo clippy -- --fix", true),
            ("cargo fmt", false),
            ("cargo fmt --check", true),
            ("cargo fmt -- --check", true),
            ("cargo fix", false),
        ];

        run_cases(&cases);
    }

    // ── Unconditional rejections ──────────────────────────────────

    /// Tests that ALL entries in the production [`MUTATING_COMMANDS`] constant
    /// are rejected. Iterates the constant directly to prevent coverage drift
    /// when entries are added or removed.
    #[test]
    fn all_mutating_commands_rejected() {
        assert_all_rejected(MUTATING_COMMANDS, |cmd| format!("{cmd} arg"));
    }

    /// Tests that all git branch mutation flags are rejected via
    /// [`check_git_subcommand_mutation`].
    #[test]
    fn git_branch_mutation_flags_rejected() {
        assert_all_rejected(GIT_BRANCH_MUTATIONS, |flag| {
            format!("git branch {flag} feature")
        });
    }

    /// Tests that all git tag mutation flags are rejected via
    /// [`check_git_subcommand_mutation`].
    #[test]
    fn git_tag_mutation_flags_rejected() {
        assert_all_rejected(GIT_TAG_MUTATIONS, |flag| format!("git tag {flag} v1.0"));
    }

    /// Tests that all git remote mutation verbs are rejected via
    /// [`check_git_subcommand_mutation`].
    #[test]
    fn git_remote_mutation_verbs_rejected() {
        assert_all_rejected(GIT_REMOTE_MUTATIONS, |verb| {
            format!("git remote {verb} origin")
        });
    }

    // ── Flag-dependent tests ──────────────────────────────────────

    #[test]
    fn flag_dependent_tests() {
        let cases = [
            // sed
            ("sed 's/a/b/' file", true),
            ("sed -i 's/a/b/' file", false),
            ("sed -i.bak 's/a/b/' file", false),
            // awk
            ("awk '{print $1}' file", true),
            ("awk -i inplace '{print $1}' file", false),
            // dd
            ("dd if=/dev/zero bs=1 count=10", true),
            ("dd if=/dev/zero of=file bs=1 count=10", false),
            // curl
            ("curl https://example.com", true),
            ("curl -o file https://example.com", false),
            ("curl -O https://example.com/file", false),
            // tar
            ("tar -tf archive.tar.gz", true),
            ("tar -xzf archive.tar.gz", false),
            ("tar -czf archive.tar.gz dir/", false),
            ("tar --list -f archive.tar.gz", true),
            // base64
            ("base64 -d file.txt", true),
            ("base64 -d -o out.bin file.txt", false),
            ("base64 --decode --output out.bin file.txt", false),
        ];

        run_cases(&cases);
    }

    // ── Chained commands ───────────────────────────────────────────

    #[test]
    fn chained_commands() {
        let cases = [
            ("cargo check && cargo test", true),
            ("cargo check && rm file", false),
            ("git status && cargo fmt", false),
            ("git log --oneline | head -20", true),
            ("cargo check; rm file", false),
        ];

        run_cases(&cases);
    }

    // ── Redirect tests ─────────────────────────────────────────────

    #[test]
    fn redirect_tests() {
        let cases = [
            // Original redirect tests
            ("echo hello > file.txt", false),
            ("echo hello > /dev/null", true),
            ("echo hello > /tmp/output.txt", true),
            ("cmd 2>&1", true),
            ("echo \"hello > world\"", true),
            ("echo hello >> /tmp/log", true),
            ("echo hello >| /tmp/force", true),
            ("cargo build > /dev/null 2>&1", true),
            // /var/tmp redirect tests
            ("echo hello > /var/tmp/output.txt", true),
            ("echo hello >> /var/tmp/log", true),
            // Redirect operators refactor tests
            ("cmd > output.txt", false),
            ("cmd 1>&2", true),
            ("cmd 2> /tmp/errors.log", true),
            ("cmd 2> errors.log", false),
            ("cmd >&2", false),
            ("echo \\> /tmp/file", true),
            ("echo \\>", true),
            ("echo \\\\\\> file", true),
            ("echo \"> /tmp/foo", true),
            ("echo '> /tmp/foo", true),
        ];

        run_cases(&cases);
    }

    // ── Heredoc quote-state tracking ────────────────────────────

    /// Tests that `<<` inside quotes is not treated as a heredoc start
    /// (fix for mahbot-73).  Without quote-state tracking in
    /// [`strip_heredoc_bodies`], a quoted `<<` would cause everything
    /// after it — including real unquoted redirect operators — to be
    /// stripped from the redirect scan string, creating a false-negative
    /// security bypass.
    #[test]
    fn heredoc_quote_state() {
        let cases = [
            // Primary bug scenario: `<<` inside single quotes followed by
            // a real redirect on the same line.  strip_heredoc_bodies must
            // NOT strip `> output.txt` because `<<` is inside quotes.
            ("echo '<<EOF' > output.txt", false),
            // Same with double quotes
            ("echo \"<<EOF\" > output.txt", false),
            // Quoted << without redirect — should be allowed regardless
            ("echo '<<EOF'", true),
            ("echo \"<<EOF\"", true),
            // <<- with dash inside single quotes, redirect follows
            ("echo '<<-EOF' > output.txt", false),
            // No-redirect variant: quoted << with no redirect (just text)
            ("echo 'before <<EOF after'", true),
            ("echo \"before <<EOF after\"", true),
            // Backslash-escaped << (double-escape)
            ("echo \\<\\<file > /etc/output", false),
            // Backslash-escaped << (single-escape)
            ("echo \\<<EOF > /etc/output", false),
            // Escaped single quote
            ("echo \\'hello > /etc/output", false),
            // Nested quotes: single-quoted string inside double quotes
            ("echo \"'<<EOF'\" > /etc/output", false),
            // Existing real heredoc behaviors still work:
            (
                "cat > /tmp/test_match.rs << 'EOF'\nfn test() { match x { \"a\" => 1, _ => 0 } }\nEOF",
                true,
            ),
            // Real heredoc with no redirect
            ("cat <<EOF\nbody\nEOF", true),
        ];

        run_cases(&cases);
    }

    // ── mktemp (temp dir, allowed) ────────────────────────────────

    #[test]
    fn mktemp_allowed() {
        let cases = [("mktemp", true), ("mktemp -t mahbot.XXXXXX", true)];

        run_cases(&cases);
    }

    // ── Prefix stripping (P0) ──────────────────────────────────────

    /// Tests that ALL delegating shell prefixes (those that forward their
    /// arguments as a command) correctly dispatch commands for read-only
    /// validation. Excludes non-delegating builtins (`cd`, `pushd`, `popd`,
    /// `export`, `source`, `.`).
    ///
    /// Three command scenarios are tested for every prefix:
    /// - `rm file` — a mutating command that must be rejected.
    /// - `git push` — a mutating git subcommand that must be rejected
    ///   (ensuring no prefix masks the git command word).
    /// - `git status` — a safe git command that must be allowed.
    #[test]
    fn shell_prefixes_delegating() {
        let cases = [
            ("rm file", false),
            ("git push", false),
            ("git status", true),
        ];

        for prefix in SHELL_PREFIXES {
            if NON_DELEGATING_PREFIXES.contains(prefix) {
                continue;
            }
            for &(command, allowed) in &cases {
                let cmd = format!("{prefix} {command}");
                if allowed {
                    ok(&cmd);
                } else {
                    assert_rejected(&cmd);
                }
            }
        }
    }

    // ── Prefix / env stripping regression tests (P0) ──────────────

    #[test]
    fn prefix_bypass_and_env() {
        let cases = [
            // Prefix stripping with flags
            ("sudo -E rm file", false),
            ("sudo git status", true),
            ("sudo cargo check", true),
            // Git prefix bypass
            ("sudo git push", false),
            ("env git push", false),
            ("GIT_DIR=/tmp sudo git push", false),
            ("sudo git stash list", true),
            ("cd", true),
            ("cd ..", true),
            // VAR=val stripping
            ("FOO=bar rm file", false),
            ("VAR=val sudo rm -rf /", false),
            ("GIT_DIR=/tmp git status", true),
        ];

        run_cases(&cases);
    }

    // ── Script interpreters & container tools: read-only usage (not blocked) ─

    #[test]
    fn script_and_container_tools() {
        let cases = [
            // Script interpreters
            ("python3 --version", true),
            ("python3 -c \"print('hello')\"", true),
            ("node -e \"console.log('hi')\"", true),
            ("bash -c \"echo hello\"", true),
            // Container tools
            ("docker ps", true),
            ("kubectl get pods", true),
        ];

        run_cases(&cases);
    }

    // ── extract_git_subcommand unit tests ──────────────────────────

    #[test]
    fn test_extract_git_subcommand() {
        struct Case {
            name: &'static str,
            input: &'static str,
            expected: &'static str,
        }

        let cases = [
            Case {
                name: "basic",
                input: "git status",
                expected: "status",
            },
            Case {
                name: "with global flag",
                input: "git -C /repo diff",
                expected: "diff",
            },
            Case {
                name: "with config",
                input: "git -c user.name=me log",
                expected: "log",
            },
            Case {
                name: "with git dir",
                input: "git --git-dir /repo status",
                expected: "status",
            },
            Case {
                name: "env assignment",
                input: "GIT_DIR=/tmp git status",
                expected: "status",
            },
            Case {
                name: "no git",
                input: "cargo build",
                expected: "",
            },
            Case {
                name: "git only",
                input: "git",
                expected: "",
            },
            Case {
                name: "full subcommand",
                input: "git branch -d feature",
                expected: "branch -d feature",
            },
            Case {
                name: "with double dash",
                input: "git -- diff",
                expected: "diff",
            },
            Case {
                name: "stash list",
                input: "git stash list",
                expected: "stash list",
            },
            Case {
                name: "multiple env",
                input: "CC=gcc CXX=g++ git status",
                expected: "status",
            },
            Case {
                name: "multiple flags",
                input: "git -C /repo --git-dir /other status",
                expected: "status",
            },
            Case {
                name: "with sudo skipped",
                input: "sudo git status",
                expected: "status",
            },
            Case {
                name: "with env skipped",
                input: "env git status",
                expected: "status",
            },
            Case {
                name: "env and sudo",
                input: "GIT_DIR=/tmp sudo git status",
                expected: "status",
            },
            Case {
                name: "sudo push",
                input: "sudo git push",
                expected: "push",
            },
            Case {
                name: "flag with multiple args",
                input: "git branch --merged master",
                expected: "branch --merged master",
            },
        ];

        for case in &cases {
            assert_eq!(
                extract_git_subcommand(case.input),
                case.expected,
                "case: {}",
                case.name
            );
        }
    }

    // ── Temp / scratch directory tests ─────────────────────────────

    #[test]
    fn temp_scratch_tests() {
        let cases = [
            (
                "cat > /tmp/test_match.rs << 'EOF'\nfn test() { match x { \"a\" => 1, _ => 0 } }\nEOF",
                true,
            ),
            ("echo hello > /private/tmp/mahbot_test_out.txt", true),
            ("tee /tmp/scratch.log", true),
            ("touch /tmp/scratch.txt", true),
            ("mkdir -p /tmp/scratch_dir", true),
            ("tee output.log", false),
            ("rm /tmp/scratch.txt", false),
            // ── Multiple path arguments (mahbot-396 security bypass) ──
            // Multiple path args under temp → should be allowed
            ("tee /tmp/scratch.log /tmp/out.txt", true),
            ("touch /tmp/a.txt /tmp/b.txt", true),
            // Mixed: one temp, one non-temp → should be rejected
            ("tee /tmp/scratch.log /etc/passwd", false),
            ("touch /tmp/scratch.txt /etc/cron.d/evil", false),
            ("mkdir -p /tmp/dir /etc/cron.d", false),
            // Mixed with redirects → only path args checked
            ("tee /tmp/scratch.log /etc/passwd > /dev/null", false),
            ("tee /tmp/scratch.log /tmp/out.txt > /dev/null", true),
            // Combined redirect tokens (2>/dev/null style)
            ("tee /tmp/scratch.log /etc/passwd 2>/dev/null", false),
            ("tee /tmp/scratch.log /tmp/out.txt 2>&1", true),
            // Heredoc with scratch mutator → heredoc body not treated as path
            ("tee /tmp/scratch.log << 'EOF'\nbody\nEOF", true),
            (
                "tee /tmp/scratch.log /tmp/out.txt << 'EOF'\nbody\nEOF",
                true,
            ),
            // 1> standalone redirect (separate target) → not a path arg
            ("tee /tmp/scratch.log 1>/dev/null", true),
            // Bash &> combined redirect → not collected as path arg
            ("tee /tmp/scratch.log &>/dev/null", true),
            ("tee /tmp/scratch.log &>>/dev/null", true),
            // 1> with space-separated target → redirect target not collected as path
            ("tee /tmp/scratch.log 1> /dev/null", true),
            // Generic digit-prefixed redirects ({digit}> and {digit}<)
            ("tee /tmp/scratch.log 3> /dev/null", true),
            ("tee /tmp/scratch.log 3< /dev/null", true),
            // Digit-prefixed heredoc (e.g. 3<<EOF) → body not treated as path
            ("tee /tmp/scratch.log 3<< 'EOF'\nbody\nEOF", true),
            // Multi-digit fd redirect with space-separated target (10> /dev/null)
            ("tee /tmp/scratch.log 10> /dev/null", true),
            // &> standalone redirect with space before target
            ("tee /tmp/scratch.log &> /dev/null", true),
            ("tee /tmp/scratch.log &>> /dev/null", true),
        ];

        run_cases(&cases);
    }

    #[test]
    fn scratch_mutators_are_subset_of_mutating_commands() {
        for cmd in SCRATCH_MUTATORS {
            assert!(
                MUTATING_COMMANDS.contains(cmd),
                "SCRATCH_MUTATORS entry '{cmd}' must also be in MUTATING_COMMANDS"
            );
        }
    }
}
