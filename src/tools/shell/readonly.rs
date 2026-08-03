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
    "shred",
    "mkfifo",
    "mknod",
    "ln",
    "install",
    "truncate",
    "fallocate",
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
    "whatchanged",
    "reflog",
    "range-diff",
    "request-pull",
    "worktree list",
    "config --list",
    "config --get",
    "config --get-all",
    "hash-object",
    "stripspace",
    "remote",
    "branch",
    "tag",
    "stash list",
    "stash show",
    "show-ref",
    "ls-remote",
];

/// Safe cargo subcommands that only affect build artifacts in `target/` or are purely
/// read-only queries. Commands that modify source files or `Cargo.lock` must NOT be added
/// here — they should get targeted rejection messages with tailored suggestions instead.
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
    "version",
    "verify-project",
    "read-manifest",
    "help",
    "bench",
];

// ── Redirect detection ───────────────────────────────────────────────────

///   Advance `i` past the current line (to the start of the next line or end of string).
fn skip_to_next_line(i: &mut usize, chars: &[(usize, char)]) {
    while *i < chars.len() && chars[*i].1 != '\n' {
        *i += 1;
    }
    if *i < chars.len() {
        *i += 1;
    }
}

/// True when `<<` at byte index `i` (outside quotes) starts a heredoc whose
/// body must be stripped. Recognizes bare (`<<EOF`, `<<-EOF`) and fd-prefixed
/// (`3<<EOF`, `1<<-EOF`) heredocs.
///
/// `<<<` is a **herestring**, not a heredoc: it has no body to strip, and
/// treating it as a heredoc would swallow everything after it — including a
/// real workspace redirect on the same line (a live-reproduced guard bypass).
fn is_heredoc_start(chars: &[(usize, char)], i: usize) -> bool {
    let bare = chars[i].1 == '<' && chars.get(i + 1).is_some_and(|(_, c)| *c == '<');
    let fd_prefixed = chars[i].1.is_ascii_digit()
        && chars.get(i + 1).is_some_and(|(_, c)| *c == '<')
        && chars.get(i + 2).is_some_and(|(_, c)| *c == '<');
    if !bare && !fd_prefixed {
        return false;
    }
    // Exclude herestrings: `<<<` (bare) and `{digit}<<<` (fd-prefixed).
    let herestring_start = if fd_prefixed { i + 3 } else { i + 2 };
    chars.get(herestring_start).is_none_or(|(_, c)| *c != '<')
}

/// True when `<<<` at byte index `i` (outside quotes) starts a herestring
/// (which has no body to strip, but whose operator must be skipped so its
/// second `<` isn't misread as a heredoc start). Recognizes bare (`<<<`)
/// and fd-prefixed (`3<<<`) herestrings.
fn is_herestring_start(chars: &[(usize, char)], i: usize) -> bool {
    let bare = chars[i].1 == '<'
        && chars.get(i + 1).is_some_and(|(_, c)| *c == '<')
        && chars.get(i + 2).is_some_and(|(_, c)| *c == '<');
    let fd_prefixed = chars[i].1.is_ascii_digit()
        && chars.get(i + 1).is_some_and(|(_, c)| *c == '<')
        && chars.get(i + 2).is_some_and(|(_, c)| *c == '<')
        && chars.get(i + 3).is_some_and(|(_, c)| *c == '<');
    bare || fd_prefixed
}

/// True when the line starting at `line_start` is the heredoc terminator line
/// for `delimiter`. For `<<-` heredocs, ALL leading TABs are stripped before
/// matching (bash strips every leading tab, not just one); for regular
/// heredocs no leading whitespace is allowed. The delimiter must be followed
/// by end-of-input, CR, or LF.
fn heredoc_terminator_matches(
    command: &str,
    line_start: usize,
    delimiter: &str,
    strip_tabs: bool,
) -> bool {
    let rest = &command[line_start..];
    let candidate = if strip_tabs {
        rest.trim_start_matches('\t')
    } else {
        rest
    };
    match candidate.strip_prefix(delimiter) {
        Some(after) => after.is_empty() || after.starts_with('\r') || after.starts_with('\n'),
        None => false,
    }
}

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
/// Two regions must REMAIN scanned so a write command cannot hide:
/// - the tail of the delimiter line (`cat <<EOF > workspace_file` writes to
///   `workspace_file` on the delimiter line itself), and
/// - the line(s) after the terminator (a write command on the line following
///   `EOF` must not escape scanning).
///
/// The heredoc marker itself is replaced with a space so the command string
/// stays well-formed for redirect scanning and newline segmentation.
///
/// Command substitutions inside **unquoted** heredoc bodies are ALSO emitted
/// (as `$(...)`/backtick spans) so they remain scanned: real bash executes
/// `$(touch x)` inside `cat <<EOF\n$(touch x)\nEOF`, so stripping the body
/// entirely would hide the mutation. Bodies of **quoted** delimiters
/// (`<<'EOF'`, `<<"EOF"`) are literal in bash — nothing is emitted for them.
///
/// Multiple heredocs declared on the same delimiter line (`<<A <<B`) are
/// tracked in declaration order; bodies are consumed in input order (bash
/// reads heredoc bodies in the order the terminators appear).
///
/// # Known limitation (pre-existing, not addressed here)
///
/// - Heredoc bodies that contain the delimiter within quotes are not detected
///   (the body-skipping loop checks for literal delimiter matches).  In a real
///   shell, a quoted delimiter in the body does NOT terminate the heredoc.
///   This can produce false negatives (allowing a dangerous redirect inside a
///   heredoc body whose delimiter appears inside quotes earlier in the body),
///   but such multi-line engineered inputs are unlikely in practice.
#[allow(clippy::too_many_lines)] // security-critical heredoc state machine
pub(super) fn strip_heredoc_bodies(command: &str) -> String {
    let chars: Vec<(usize, char)> = command.char_indices().collect();
    let mut out = String::with_capacity(command.len());
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut i = 0;

    // Pending heredocs declared on the current delimiter line, in declaration
    // order: (delimiter, strip-tabs flag, body-expands flag). The last field
    // is true when the delimiter was unquoted — bash then performs command
    // substitution inside the body, so substitution spans must be emitted.
    let mut queue: Vec<(String, bool, bool)> = Vec::new();
    // When true, the delimiter line is fully emitted and body lines are being
    // skipped until the front of `queue` matches.
    let mut skipping_body = false;

    while i < chars.len() {
        // ── Heredoc body skipping ─────────────────────────────────
        // Skip whole lines until the line matches the front heredoc's
        // terminator. The delimiter-line tail was already emitted.
        if !queue.is_empty() && skipping_body {
            let line_start = chars[i].0;
            if heredoc_terminator_matches(command, line_start, &queue[0].0, queue[0].1) {
                // Advance to the terminator's line ending so the newline is
                // emitted as a command separator and the line AFTER the
                // terminator remains scanned (a write command on the next
                // line must not escape scanning). The terminator may be
                // preceded by any number of tabs under `<<-` and its
                // delimiter may contain shell metacharacters (quoted
                // delimiters), so skip to the line end rather than assuming
                // the delimiter starts at `line_start`.
                let line_end = command[line_start..]
                    .find('\n')
                    .map_or(command.len(), |off| line_start + off);
                i = chars
                    .iter()
                    .position(|(byte, _)| *byte >= line_end)
                    .unwrap_or(chars.len());
                queue.remove(0);
                if queue.is_empty() {
                    skipping_body = false;
                }
                continue;
            }
            // Emit command-substitution spans from unquoted bodies so they
            // stay scanned (see the security invariant above).
            if queue[0].2 {
                let line_end = command[line_start..]
                    .find('\n')
                    .map_or(command.len(), |off| line_start + off);
                if emit_body_substitutions(&command[line_start..line_end], &mut out)
                    && line_end < command.len()
                {
                    // Keep emitted spans on separate lines so segmentation
                    // treats each as its own command.
                    out.push('\n');
                }
            }
            skip_to_next_line(&mut i, &chars);
            continue;
        }

        // ── Delimiter-line tail ──────────────────────────────────
        // After the heredoc marker, the rest of the delimiter line remains
        // scanned (a redirect on the delimiter line is a real write target),
        // and additional `<<` on the same line start further heredocs.
        if !queue.is_empty() {
            if chars[i].1 == '\n' {
                out.push('\n');
                i += 1;
                skipping_body = true;
                continue;
            }
            if !super::track_char_context(chars[i].1, &mut in_single, &mut in_double, &mut escaped)
            {
                out.push(chars[i].1);
                i += 1;
                continue;
            }
            if is_herestring_start(&chars, i) {
                i = emit_herestring(&chars, i, &mut out);
                continue;
            }
            if is_heredoc_start(&chars, i) {
                match consume_heredoc_marker(command, &chars, i, &mut out, &mut queue) {
                    Some(next) => i = next,
                    None => break, // dangling `<<` at end of input
                }
                continue;
            }
            out.push(chars[i].1);
            i += 1;
            continue;
        }

        // ── Normal scanning (with heredoc detection) ─────────────
        if !super::track_char_context(chars[i].1, &mut in_single, &mut in_double, &mut escaped) {
            out.push(chars[i].1);
            i += 1;
            continue;
        }

        if is_herestring_start(&chars, i) {
            i = emit_herestring(&chars, i, &mut out);
            continue;
        }

        if is_heredoc_start(&chars, i) {
            match consume_heredoc_marker(command, &chars, i, &mut out, &mut queue) {
                Some(next) => i = next,
                None => break, // dangling `<<` at end of input — nothing to strip
            }
            continue;
        }

        out.push(chars[i].1);
        i += 1;
    }

    out
}

/// Consume a heredoc marker at `i` (bare `<<` or fd-prefixed `{digit}<<`):
/// parse its delimiter and push `(delimiter, strip_tabs, expands)` onto
/// `queue`. The marker itself is replaced by a single space in `out` so the
/// tail of the delimiter line remains scannable (a redirect there is a real
/// write target). Returns `Some(next_index)` or `None` for a dangling `<<`
/// at end of input.
///
/// Used by both the normal-scanning path and the delimiter-line-tail path so
/// the marker grammar (fd prefix, `<<-` tabs, whitespace, quoted delimiter)
/// lives in exactly one place.
fn consume_heredoc_marker(
    command: &str,
    chars: &[(usize, char)],
    mut i: usize,
    out: &mut String,
    queue: &mut Vec<(String, bool, bool)>,
) -> Option<usize> {
    out.push(' ');
    if chars[i].1.is_ascii_digit() {
        i += 1; // fd-prefixed heredoc (`3<<EOF`)
    }
    i += 2;
    if i >= chars.len() {
        return None; // dangling `<<` at end of input — nothing to strip
    }
    while i < chars.len() && chars[i].1.is_whitespace() {
        i += 1;
    }
    let mut strip_tabs = false;
    if i < chars.len() && chars[i].1 == '-' {
        strip_tabs = true;
        i += 1;
    }
    while i < chars.len() && chars[i].1.is_whitespace() {
        i += 1;
    }
    let (delimiter, delim_end, quoted) = parse_heredoc_delimiter(command, chars[i].0);
    queue.push((delimiter, strip_tabs, !quoted));
    Some(
        chars
            .iter()
            .position(|(byte, _)| *byte >= delim_end)
            .unwrap_or(chars.len()),
    )
}

/// Emit a herestring operator (`<<<` or `{digit}<<<`) unchanged so the token
/// classifier still sees it and can skip the herestring content word as a
/// path argument (a herestring has no body to strip).
fn emit_herestring(chars: &[(usize, char)], i: usize, out: &mut String) -> usize {
    let skip = if chars[i].1.is_ascii_digit() { 4 } else { 3 };
    for c in &chars[i..i + skip] {
        out.push(c.1);
    }
    i + skip
}

/// Emit command-substitution spans (`$(...)`, backticks) found in an
/// unquoted heredoc body line, so they remain scanned as nested commands.
///
/// Real bash performs command substitution inside heredoc bodies whose
/// delimiter is unquoted (`cat <<EOF\n$(touch x)\nEOF` runs `touch x`), so
/// stripping the body entirely would hide the mutation from the guard.
/// Quoted delimiters (`<<'EOF'`, `<<"EOF"`) make the body literal — no
/// expansion occurs — so this is only called for unquoted bodies.
/// Backslash escapes (`\$`, `` \` ``, `\\`) prevent expansion and are
/// skipped. Returns `true` when at least one span was emitted.
fn emit_body_substitutions(line: &str, out: &mut String) -> bool {
    let mut emitted = false;
    let bytes = line.as_bytes();
    let mut j = 0;
    while j < line.len() {
        let c = line[j..].chars().next().expect("j < line.len()");
        if c == '\\' {
            // Backslash escapes the next char in heredoc bodies — the
            // escaped char is literal, so skip past both.
            j += c.len_utf8();
            if j < line.len() {
                j += line[j..].chars().next().expect("j < line.len()").len_utf8();
            }
            continue;
        }
        if c == '$' && bytes.get(j + 1) == Some(&b'(') {
            let (_, next) = find_substitution_end(line, j + 2);
            out.push_str(&line[j..next]);
            j = next;
            emitted = true;
            continue;
        }
        if c == '`' {
            let (_, next) = find_backtick_end(line, j + 1);
            out.push_str(&line[j..next]);
            j = next;
            emitted = true;
            continue;
        }
        j += c.len_utf8();
    }
    emitted
}

/// Parse a heredoc delimiter token starting at `start` (byte index).
/// Returns `(delimiter, end_byte, was_quoted)`. `was_quoted` is true when
/// the delimiter was quoted (`<<'EOF'`, `<<"EOF"`) — bash then treats the
/// body as literal (no parameter/command/arithmetic expansion).
fn parse_heredoc_delimiter(command: &str, start: usize) -> (String, usize, bool) {
    let rest = &command[start..];
    if let Some(rest) = rest.strip_prefix('\'') {
        if let Some(end) = rest.find('\'') {
            let delim = &rest[..end];
            return (delim.to_string(), start + 1 + end + 1, true);
        }
    } else if let Some(rest) = rest.strip_prefix('"')
        && let Some(end) = rest.find('"')
    {
        let delim = &rest[..end];
        return (delim.to_string(), start + 1 + end + 1, true);
    }

    let end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
    (rest[..end].to_string(), start + end, false)
}

/// Detect output redirect operators in a segment, respecting quote state.
/// Returns an `Err` (rejection message) if the segment contains a redirect
/// that writes to a non-allowed destination (not `/dev/null`, not temp).
///
/// Runs per-segment so the redirect scan sees the tracked current directory
/// and environment bindings at the segment's position in the command chain.
/// The segment is expected to already be heredoc-stripped (see
/// [`strip_heredoc_bodies`]). Command substitutions (`$(...)`, backticks) are
/// skipped here — their contents are validated separately by
/// [`scan_substitutions`], and re-scanning them would double-validate
/// (e.g. a `2>/dev/null` inside a substitution would be misread with the
/// substitution's closing delimiter attached to the target). Substitution
/// starts are recognized even inside double quotes: bash executes
/// `"$(touch f)"`, so a write hidden in a double-quoted substitution must
/// still be skipped here and validated by [`scan_substitutions`].
fn has_disallowed_redirect(segment: &str, state: &ValidationState) -> Result<(), String> {
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let bytes = segment.as_bytes();

    let mut i = 0;
    while i < segment.len() {
        let c = segment[i..].chars().next().expect("i < len");
        // Command substitutions run inside double quotes too (`"$(...)"`,
        // `` "`...`" ``), so detect their starts before the quote-state skip
        // below. Inside single quotes `$(`/backticks are literal; a preceding
        // escape backslash makes them literal too.
        if !escaped && !in_single && (c == '$' && bytes.get(i + 1) == Some(&b'(') || c == '`') {
            let next = if c == '$' {
                let (_, next) = find_substitution_end(segment, i + 2);
                next
            } else {
                let (_, next) = find_backtick_end(segment, i + 1);
                next
            };
            i = next;
            continue;
        }
        // [`super::track_char_context`] handles both backslash escaping
        // and quote state transitions, returning `false` when the
        // character should be skipped for redirect detection (escaped,
        // backslash, quote char, or inside quotes).
        if !super::track_char_context(c, &mut in_single, &mut in_double, &mut escaped) {
            i += c.len_utf8();
            continue;
        }

        // Check for 2>&1 and 1>&2 — pure stderr-to-stdout merges, always
        // allowed.  These are 4-character patterns.
        if segment[i..].starts_with("2>&1") || segment[i..].starts_with("1>&2") {
            i += 4;
            continue;
        }

        // 2-character redirect operators
        let redirect_len = if segment[i..].starts_with(">&")
            || segment[i..].starts_with(">>")
            || segment[i..].starts_with(">|")
            || segment[i..].starts_with("2>")
        {
            2
        } else if c == '>' {
            1
        } else {
            i += c.len_utf8();
            continue;
        };
        i += redirect_len;

        // Extract target after redirect operator. `)` and `}` terminate the
        // target so a suppressed-stderr read inside a substitution
        // (`2>/dev/null)`) or a brace-closed group (`2>/dev/null}`) is
        // recognized as targeting /dev/null instead of being rejected.
        let after = &segment[i..].trim_start();
        let target = after
            .split(|ch: char| ch.is_whitespace() || matches!(ch, '&' | ';' | '|' | ')' | '}'))
            .next()
            .unwrap_or("");

        if target.is_empty() {
            // No target — bare redirect, reject
            return Err(format!(
                "⚠️ Read-only mode: command contains a disallowed output redirect.\n\
                 Command: `{segment}`\n\
                 Redirects are only allowed to /dev/null, 2>&1, 1>&2, or paths under /tmp, /var/tmp, or the OS temp directory.\n\
                 Suggestion: pipe to a pager (e.g., `| less`) or use `| head` to limit output."
            ));
        }

        // Allowed targets
        if target == "/dev/null" {
            continue;
        }

        let resolved = resolve_path_word(target, state);
        let allowed = resolved.is_some_and(|path| is_path_under_temp(&path, state.ctx));
        if !allowed {
            // Absolute/relative non-temp non-devnull = disallowed
            return Err(format!(
                "⚠️ Read-only mode: command contains a disallowed output redirect.\n\
                 Command: `{segment}`\n\
                 Redirects are only allowed to /dev/null, 2>&1, 1>&2, or paths under /tmp, /var/tmp, or the OS temp directory.\n\
                 Suggestion: pipe to a pager (e.g., `| less`) or use `| head` to limit output."
            ));
        }
    }

    Ok(())
}

/// Resolve a path word (mutator argument, redirect target, or flag value)
/// into an absolute path against the tracked state: shell-variable expansion
/// (`$VAR`/`${VAR}`), balanced surrounding quotes, tilde handling, and
/// resolution against the tracked current directory. Returns `None` when the
/// word cannot be safely resolved — unknown/poisoned variables, tilde paths
/// (never under temp), unbalanced/mixed quotes, relative paths without a
/// tracked CWD, or relative globs — all of which reject (fail-closed).
fn resolve_path_word(word: &str, state: &ValidationState) -> Option<std::path::PathBuf> {
    // Strip balanced surrounding quotes (`"/tmp/x"`, `'/tmp/x'`).
    // Single-quoted words do NOT expand variables (`'$TMPDIR/x'` is a literal
    // filename containing `$`, which never resolves under temp).
    let (content, single_quoted) = strip_outer_quotes(word)?;
    if content.is_empty() {
        return None;
    }
    // Tilde paths resolve to $HOME (or another user's home) — never under a
    // temp root (decision 6), so they always reject.
    if content.starts_with('~') {
        return None;
    }
    let expanded = expand_vars(content, single_quoted, state)?;
    if expanded.is_empty() {
        return None;
    }
    let path = std::path::PathBuf::from(&expanded);
    if path.is_relative() {
        // Relative globs stay rejected (they could match workspace files).
        if contains_glob(&expanded) {
            return None;
        }
        let cwd = state.cwd.as_ref()?; // fail-closed when tracking was reset
        return Some(cwd.join(&expanded));
    }
    Some(path)
}

/// Strip balanced surrounding single/double quotes from a shell word.
/// Returns `(content, was_single_quoted)`, or `None` for unbalanced/mixed
/// quotes (e.g. `"/tmp`, `'/tmp"`, `ab"cd`).
fn strip_outer_quotes(word: &str) -> Option<(&str, bool)> {
    let bytes = word.as_bytes();
    if word.len() >= 2
        && matches!(bytes[0], b'\'' | b'"')
        && matches!(bytes[word.len() - 1], b'\'' | b'"')
    {
        let open = bytes[0];
        let close = bytes[word.len() - 1];
        if open != close {
            return None; // mixed quotes: `'..."`
        }
        let inner = &word[1..word.len() - 1];
        if inner.contains(open as char) {
            return None; // unbalanced inside: `"a"b"`
        }
        return Some((inner, open == b'\''));
    }
    if word.contains(['\'', '"']) {
        return None;
    }
    Some((word, false))
}

/// Expand `$VAR` / `${VAR}` references in `word`. Inside single quotes nothing
/// expands. Returns `None` (reject) when the expansion is unprovable: an
/// unbound variable without a temp anchor, a poisoned variable, `$HOME`,
/// `$PWD`-when-untracked, or a `..` escape segment after an opaque suffix.
///
/// Temp-tagged variables (`NAME=$(mktemp -d)`) expand to a synthetic anchor
/// path (first allowed temp root + one level — the same depth as a real
/// `mktemp` result, so `..` chains escape at the same depth). The `$RANDOM`
/// builtin after a temp anchor expands to an opaque placeholder segment
/// (`$SNAP/$RANDOM/f`); any other unbound variable fails closed (its ambient
/// value is outside the guard's tracking model). The tail after the first
/// opaque segment is unverifiable, so any `..` path segment there fails
/// closed.
fn expand_vars(word: &str, single_quoted: bool, state: &ValidationState) -> Option<String> {
    if single_quoted {
        // No expansion inside single quotes — a literal `$` is not a valid
        // temp path.
        if word.contains('$') {
            return None;
        }
        return Some(word.to_string());
    }
    let mut out = String::with_capacity(word.len());
    let mut opaque_from: Option<usize> = None;
    let mut i = 0;
    while i < word.len() {
        let rest = &word[i..];
        // Escaped dollar/backslash/backtick: literal character, no expansion.
        if let Some(next) = rest.strip_prefix('\\') {
            let c = next.chars().next().expect("non-empty after backslash");
            if matches!(c, '$' | '\\' | '`') {
                out.push(c);
            } else {
                out.push('\\');
                out.push(c);
            }
            i += 1 + c.len_utf8();
            continue;
        }
        if let Some((name, len)) = parse_var_ref(rest) {
            match resolve_var(name, state) {
                VarValue::Concrete(text) => out.push_str(&text),
                VarValue::TempRoot => out.push_str(&temp_anchor_path(state.ctx)),
                VarValue::Opaque => {
                    // Only the `$RANDOM` builtin is a safe opaque suffix after
                    // a temp anchor: it always expands to a bare word (or
                    // empty in POSIX sh), never to a path that could escape.
                    // Any other unbound variable fails closed — its ambient
                    // value is outside the guard's tracking model.
                    if name != "RANDOM" {
                        return None;
                    }
                    if opaque_from.is_none() && !is_under_temp_prefix(&out, state) {
                        return None;
                    }
                    if opaque_from.is_none() {
                        opaque_from = Some(out.len());
                    }
                    out.push_str(OPAQUE_SEGMENT);
                }
                VarValue::Blocked => return None,
            }
            i += len;
        } else {
            let c = rest.chars().next().expect("i < len");
            // An unescaped `$(`/backtick (command substitution) or an
            // unparseable `${...}` (parameter expansion with modifiers, e.g.
            // `${FOO:-../etc}`) has an unprovable output, so the word cannot
            // resolve to a concrete path — fail-closed (the escaped forms
            // were handled above; single-quoted words never reach this
            // branch; plain `${NAME}` parses as a variable reference).
            if c == '`'
                || (c == '$'
                    && rest
                        .as_bytes()
                        .get(1)
                        .is_some_and(|b| matches!(*b, b'(' | b'{')))
            {
                return None;
            }
            out.push(c);
            i += c.len_utf8();
        }
    }
    // The tail after the first opaque suffix is unverifiable: a `..` path
    // segment there could escape through the unknown value (fail-closed).
    if let Some(start) = opaque_from
        && out[start..].split('/').any(|seg| seg == "..")
    {
        return None;
    }
    Some(out)
}

/// Synthetic anchor path for a temp-tagged variable (`NAME=$(mktemp -d)`):
/// the first allowed temp root plus one level. The concrete value is unknown,
/// but any `mktemp` result lives exactly one level below a temp root, so
/// `..` chains over this anchor produce the same under-temp verdict as the
/// real value.
fn temp_anchor_path(ctx: &CheckContext) -> String {
    let root = ctx
        .temp_roots
        .first()
        .map_or_else(|| "/tmp".to_string(), |p| p.to_string_lossy().into_owned());
    format!("{root}/{TEMP_ANCHOR_SEGMENT}")
}

/// Placeholder segment for a temp-tagged variable's unknown concrete value.
const TEMP_ANCHOR_SEGMENT: &str = "__mahbot_temp__";

/// Placeholder segment for an unbound variable used as an opaque suffix after
/// a temp anchor. Must not contain `/` or `..`.
const OPAQUE_SEGMENT: &str = "__mahbot_opaque__";

/// True when the partial expansion text is provably under a temp root (a
/// literal temp prefix or a temp-anchored variable has been emitted).
fn is_under_temp_prefix(out: &str, state: &ValidationState) -> bool {
    let p = Path::new(out);
    p.is_absolute() && crate::tools::path::is_path_under_roots(p, &state.ctx.temp_roots)
}

/// Parse a `$NAME` or `${NAME}` reference at the start of `rest`.
/// Returns `(name, total_consumed_len)`.
fn parse_var_ref(rest: &str) -> Option<(&str, usize)> {
    let after_dollar = rest.strip_prefix('$')?;
    if let Some(braced) = after_dollar.strip_prefix('{') {
        let end = braced.find('}')?;
        let name = &braced[..end];
        if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Some((name, 2 + end + 1)); // $ { name }
        }
        return None;
    }
    let name_len = after_dollar
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .count();
    if name_len == 0 {
        return None;
    }
    Some((&after_dollar[..name_len], 1 + name_len))
}

/// How a variable reference resolves for expansion.
enum VarValue {
    /// Concrete expansion text (a fully-resolved value, e.g. `/tmp`,
    /// `$PWD`'s tracked CWD).
    Concrete(String),
    /// Unknown concrete value provably under a temp root
    /// (`NAME=$(mktemp -d)`).
    TempRoot,
    /// Unbound/unknown variable — only the `$RANDOM` builtin is usable as an
    /// opaque suffix after a temp anchor has been established
    /// (`$SNAP/$RANDOM/f`); any other unbound variable fails closed.
    Opaque,
    /// Poisoned or otherwise unprovable (`$HOME`, untracked `$PWD`) — always
    /// fail-closed.
    Blocked,
}

/// Resolve a single variable reference for expansion.
///
/// - `$PWD` resolves to the tracked CWD (initial = workspace root); `Blocked`
///   when tracking was reset (fail-closed).
/// - `$HOME` and poisoned variables are always `Blocked` (decision 6).
/// - Bound variables (TMPDIR/TMP/TEMP from the session env, plus any variable
///   assigned a temp-root value in the chain) resolve to their value or
///   [`VarValue::TempRoot`] when the value is a fresh `$(mktemp -d)` root.
/// - Unbound variables are [`VarValue::Opaque`].
fn resolve_var(name: &str, state: &ValidationState) -> VarValue {
    match name {
        "PWD" => match &state.cwd {
            Some(cwd) => VarValue::Concrete(cwd.to_string_lossy().into_owned()),
            None => VarValue::Blocked,
        },
        "HOME" => VarValue::Blocked,
        _ => match state.vars.get(name) {
            None => VarValue::Opaque,
            Some(b) if b.poisoned => VarValue::Blocked,
            Some(b) => match &b.value {
                // `value == None` marks a temp-tagged binding
                // (`NAME=$(mktemp -d)`) with an unknown concrete value.
                Some(v) => VarValue::Concrete(v.clone()),
                None => VarValue::TempRoot,
            },
        },
    }
}

/// True when the string contains glob metacharacters (`*`, `?`, `[`).
fn contains_glob(s: &str) -> bool {
    s.contains(['*', '?', '['])
}

/// True when `path` is under one of the context's allowed temp roots.
///
/// The lexical root check is followed by symlink resolution of the deepest
/// existing ancestor: a symlink inside a temp root can point at a non-temp
/// directory (e.g. the workspace), and writes through it land outside every
/// temp root — `is_dir` follows the symlink while a purely lexical root
/// comparison does not. Nonexistent tails climb to their existing parent,
/// which is where the write actually lands.
fn is_path_under_temp(path: &std::path::Path, ctx: &CheckContext) -> bool {
    if !crate::tools::path::is_path_under_roots(path, &ctx.temp_roots) {
        return false;
    }
    let mut probe = path;
    loop {
        if let Ok(canon) = std::fs::canonicalize(probe) {
            return crate::tools::path::is_path_under_roots(&canon, &ctx.temp_roots);
        }
        let Some(parent) = probe.parent() else {
            return true;
        };
        if parent == probe {
            return true;
        }
        probe = parent;
    }
}

/// Validation context for a read-only shell command: the workspace root,
/// temp roots, and session-environment snapshot of the session being
/// validated — never the daemon process's environment (see the guard
/// contract, resolved decisions 6 and 9).
#[derive(Debug, Clone)]
pub(super) struct CheckContext {
    /// Workspace root of the session being validated. Initial tracked CWD and
    /// the reference against which workspace writes are detected.
    workspace_root: std::path::PathBuf,
    /// Allowed temp roots for scratch/temp writes. Defaults to the shared
    /// path-root set; injectable in tests so a fixture workspace can be
    /// hermetic (kept distinct from the temp roots).
    temp_roots: Vec<std::path::PathBuf>,
    /// Baseline temp-variable bindings (TMPDIR/TMP/TEMP) of the session's
    /// shell environment. The daemon's shell launcher sets `TMPDIR=/tmp`
    /// (see `baseline_env_value`), so the production default is exactly that;
    /// tests inject their own.
    temp_vars: Vec<(String, String)>,
}

impl CheckContext {
    /// Context for a session rooted at `workspace_root` with the standard
    /// OS temp roots and the session shell's temp-variable bindings.
    pub(super) fn for_workspace(workspace_root: &Path) -> Self {
        Self {
            workspace_root: workspace_root.to_path_buf(),
            temp_roots: crate::tools::path::allowed_temp_roots(),
            temp_vars: vec![("TMPDIR".to_string(), "/tmp".to_string())],
        }
    }

    #[cfg(test)]
    fn for_test(
        workspace_root: &Path,
        temp_roots: Vec<std::path::PathBuf>,
        temp_vars: Vec<(String, String)>,
    ) -> Self {
        Self {
            workspace_root: workspace_root.to_path_buf(),
            temp_roots,
            temp_vars,
        }
    }
}

/// Binding state for one tracked variable.
#[derive(Debug, Clone)]
struct VarBinding {
    /// Bound value; `None` = unknown concrete value provably under a temp
    /// root (`NAME=$(mktemp -d)`).
    value: Option<String>,
    /// When true, subsequent expansions of this variable reject (fail-closed).
    /// Set by binding to a non-temp path; cleared by re-binding to a valid
    /// temp root.
    poisoned: bool,
}

/// Mutable state threaded through the validation of a single command string:
/// the tracked current directory and temp-variable bindings.
struct ValidationState<'a> {
    ctx: &'a CheckContext,
    /// Tracked current directory. `None` = fail-closed: relative paths reject
    /// (decision 7 — any non-literal-absolute `cd` form resets tracking).
    cwd: Option<std::path::PathBuf>,
    /// Tracked variable bindings by name: the session temp variables
    /// (TMPDIR/TMP/TEMP) plus any variable assigned a temp-root value
    /// (`SNAP=$(mktemp -d)`, `SNAP=/tmp/x`, ...) in the command chain.
    vars: std::collections::HashMap<String, VarBinding>,
    /// Directories provably created by an approved `mkdir` earlier in this
    /// command chain (normalized temp paths). A later `cd` into one of them
    /// tracks even when the directory did not exist at validation time.
    created_dirs: std::collections::HashSet<std::path::PathBuf>,
}

impl ValidationState<'_> {
    /// Initial state: CWD = the session's workspace root; temp vars bound from
    /// the context's session-environment snapshot.
    fn new(ctx: &CheckContext) -> ValidationState<'_> {
        let mut vars = std::collections::HashMap::new();
        for (name, value) in &ctx.temp_vars {
            vars.insert(
                name.clone(),
                VarBinding {
                    value: Some(value.clone()),
                    poisoned: false,
                },
            );
        }
        ValidationState {
            ctx,
            cwd: Some(ctx.workspace_root.clone()),
            vars,
            created_dirs: std::collections::HashSet::new(),
        }
    }

    /// Snapshot of the current tracking state. Used for command substitutions,
    /// which execute in a subshell: `cd`/export inside `$(...)` must not leak
    /// to the outer command's tracking.
    fn snapshot(&self) -> ValidationState<'_> {
        ValidationState {
            ctx: self.ctx,
            cwd: self.cwd.clone(),
            vars: self.vars.clone(),
            created_dirs: self.created_dirs.clone(),
        }
    }
}

// ── Main validation function ──────────────────────────────────────────────

/// Validate a shell command for read-only execution.
///
/// Splits chained commands into segments, checks each segment against
/// the allowlists and rejection rules, and returns `Ok(())` if the
/// command is safe, or `Err(String)` with a descriptive rejection message.
///
/// Validation is scoped to the session being validated (workspace root +
/// shell environment from `ctx`) — never the daemon process's environment.
pub(super) fn check_command(command_str: &str, ctx: &CheckContext) -> Result<(), String> {
    let trimmed = command_str.trim();
    if trimmed.is_empty() {
        return Ok(());
    }
    let mut state = ValidationState::new(ctx);
    validate_string(trimmed, &mut state)
}

/// Validate a command string (which may contain heredocs, substitutions, and
/// chained segments) against `state`.
fn validate_string(s: &str, state: &mut ValidationState) -> Result<(), String> {
    // Heredoc bodies must be removed before ANY scanning: a heredoc body is
    // literal text — never commands, redirects, or substitutions. Bodies are
    // also excluded from the shared segmentation (they never become segments).
    // Command substitutions inside *unquoted* bodies are emitted by the
    // stripper so they remain scanned (see [`strip_heredoc_bodies`]).
    let scan = strip_heredoc_bodies(s);

    // Segment once on the already-stripped string (no redundant second strip).
    let segments = super::extract_command_segments_stripped(&scan);
    for segment in &segments {
        // Output redirects are validated against the state BEFORE this
        // segment's own `cd`/export bindings: bash expands redirect targets
        // with the shell's current variables, and an env-prefix assignment
        // does not affect them.
        has_disallowed_redirect(segment, state)?;
        // Command substitutions are validated with the state as of this
        // segment — prior segments' `cd`/export bindings apply. Like
        // redirects, a substitution does not see its own segment's
        // env-prefix (it expands before the prefix is applied to the
        // command), so substitution scanning runs before `check_segment`
        // applies this segment's bindings. Nested validation snapshots the
        // state: `cd`/export inside `$(...)` runs in a subshell and must not
        // leak to the outer command.
        scan_substitutions(segment, state)?;
        check_segment(segment, state)?;
    }

    Ok(())
}

/// Validate command substitutions (`$(...)` and backticks) as nested commands.
///
/// Substitution contents are located with quote/escape awareness and run
/// through the full command validation recursively (heredocs, redirects,
/// nested substitutions, and mutator checks all apply inside a substitution).
/// Substitution starts are recognized even inside double quotes: bash executes
/// `"$(touch f)"` and `` "`touch f`" ``, so a write hidden in a double-quoted
/// substitution must be found here. Inside single quotes (and after an escape
/// backslash) `$(`/backticks are literal and are not substitutions.
///
/// Called once per segment from [`validate_string`]'s segment loop, so each
/// substitution is validated with the state at its segment's position —
/// prior segments' `cd`/export bindings apply (e.g. `export TMPDIR=/etc`
/// poisons `$TMPDIR` expansions in a later substitution). The nested
/// validation runs against a snapshot of the state because a substitution
/// executes in a subshell: `cd`/export inside `$(...)` does not affect the
/// outer command.
fn scan_substitutions(s: &str, state: &mut ValidationState) -> Result<(), String> {
    let mut i = 0;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    while i < s.len() {
        let c = s[i..].chars().next().expect("i < s.len()");
        // Detect substitution starts before the quote-state skip: they run
        // inside double quotes (`"$(...)"`, `` "`...`" ``) but are literal
        // inside single quotes or after an escape backslash.
        if !escaped
            && !in_single
            && (c == '$' && s.as_bytes().get(i + 1) == Some(&b'(') || c == '`')
        {
            if c == '$' {
                let (inner, next) = find_substitution_end(s, i + 2);
                validate_substitution_content(inner, state)?;
                i = next;
            } else {
                let (inner, next) = find_backtick_end(s, i + 1);
                validate_substitution_content(inner, state)?;
                i = next;
            }
            continue;
        }
        if !super::track_char_context(c, &mut in_single, &mut in_double, &mut escaped) {
            i += c.len_utf8();
            continue;
        }
        i += c.len_utf8();
    }
    Ok(())
}

/// Find the matching close paren for a `$(` substitution whose content starts
/// at byte `start`. Quote-aware and nesting-aware. Returns
/// `(content, index_after_close)`. When unterminated, the rest of the string
/// is returned as content (it still gets validated — fail-closed).
fn find_substitution_end(s: &str, start: usize) -> (&str, usize) {
    let bytes = s.as_bytes();
    let mut depth = 1usize;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut i = start;
    while i < s.len() {
        let c = s[i..].chars().next().expect("i < s.len()");
        if !super::track_char_context(c, &mut in_single, &mut in_double, &mut escaped) {
            i += c.len_utf8();
            continue;
        }
        if c == '(' && i > 0 && bytes[i - 1] == b'$' {
            // Nested `$(` substitution
            depth += 1;
        } else if c == ')' {
            depth -= 1;
            if depth == 0 {
                return (&s[start..i], i + 1);
            }
        }
        i += c.len_utf8();
    }
    (&s[start..], s.len())
}

/// Find the closing backtick for a command substitution starting at byte
/// `start`. Handles `\`` escapes (POSIX backtick quoting). Returns
/// `(content, index_after_close)`; unterminated backticks return the rest as
/// content (validated anyway — fail-closed).
fn find_backtick_end(s: &str, start: usize) -> (&str, usize) {
    let mut i = start;
    while i < s.len() {
        let c = s[i..].chars().next().expect("i < s.len()");
        if c == '\\' {
            i += c.len_utf8();
            if i < s.len() {
                i += s[i..].chars().next().expect("i < s.len()").len_utf8();
            }
            continue;
        }
        if c == '`' {
            return (&s[start..i], i + 1);
        }
        i += c.len_utf8();
    }
    (&s[start..], s.len())
}

/// Validate a substitution body as a nested command against a snapshot of the
/// current state (substitutions run in a subshell — state changes don't leak
/// to the outer command).
fn validate_substitution_content(inner: &str, state: &mut ValidationState) -> Result<(), String> {
    let mut snapshot = state.snapshot();
    validate_string(inner, &mut snapshot)
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

/// Temp-directory mutators allowed when all explicit path args are under temp.
///
/// These are commands that modify files on disk but are allowed in read-only mode
/// when all path arguments target temp directories (/tmp, /var/tmp, or the OS temp
/// directory). The prompt tells agents: "Writing to the OS temp directory is allowed."
const TEMP_MUTATORS: &[&str] = &[
    "cp", "mv", "rm", "rmdir", "unlink", "gzip", "gunzip", "bzip2", "xz", "zstd", "zip",
];

/// A flag-dependent command check: if the command's first word matches `verb`
/// and the `predicate` returns true, the command is rejected with the given message.
struct FlagCheck {
    verb: &'static str,
    predicate: fn(&str, &ValidationState) -> bool,
    rejection: &'static str,
    suggestion: &'static str,
}

/// Flag-dependent checks: reject commands that use mutation flags.
/// Each entry tests a specific verb + predicate combination.
const FLAG_CHECKS: &[FlagCheck] = &[
    FlagCheck {
        verb: "sed",
        predicate: has_sed_mutation,
        rejection: "`sed -i` is not allowed outside temp directories — it modifies files in-place.",
        suggestion: "use `sed` without `-i` to output to stdout, e.g. `sed 's/a/b/' file`, or use `-i` with a path under /tmp.",
    },
    FlagCheck {
        verb: "awk",
        predicate: has_inplace,
        rejection: "`awk -i inplace` is not allowed — it modifies files in-place.",
        suggestion: "use `awk` without `-i inplace` to output to stdout.",
    },
    FlagCheck {
        verb: "dd",
        predicate: has_dd_mutation,
        rejection: "`dd of=...` is not allowed outside temp directories — it writes to a file.",
        suggestion: "use `dd` without `of=` to output to stdout, or use `of=/tmp/...` to write to temp.",
    },
    FlagCheck {
        verb: "curl",
        predicate: has_curl_mutation,
        rejection: "`curl` with output flags is not allowed outside temp directories.",
        suggestion: "use `curl` without output flags to display content in stdout, or use `-o /tmp/...` to save to temp.",
    },
    // Note: `is_tar_mutating` uses negative detection (see its doc comment).
    // Unlike the five entries above, the predicate returns `true` for
    // *anything* not explicitly whitelisted as list-only.
    FlagCheck {
        verb: "tar",
        predicate: is_tar_mutating,
        rejection: "`tar` is only allowed with `-t`/`--list` (list) mode.",
        suggestion: "use `tar -tf archive.tar` to list contents.",
    },
    FlagCheck {
        verb: "base64",
        predicate: has_base64_mutation,
        rejection: "`base64 -d` with `-o` is not allowed outside temp directories — it writes decoded output to a file.",
        suggestion: "use `base64 -d` without `-o` to output to stdout, or use `-o /tmp/...` to write to temp.",
    },
    FlagCheck {
        verb: "wget",
        predicate: has_wget_mutation,
        rejection: "`wget` with output flags is not allowed outside temp directories.",
        suggestion: "use `curl` without output flags to display content in stdout, or use `wget -O /tmp/...` to save to temp.",
    },
];

/// Collect all non-flag, non-redirect, non-heredoc path-like arguments from a
/// command segment, scanning the **original** whitespace-split tokens.
///
/// This replaces the previous implementation that used [`canonical_command`],
/// which truncated to the first non-flag argument only, meaning multiple
/// path arguments (e.g. `tee /tmp/a /etc/passwd`) had only the first one
/// validated — a security bypass.
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
///
/// Heredoc markers are classified as self-contained redirects: the segments
/// passed here are always heredoc-stripped by [`strip_heredoc_bodies`] (the
/// marker is replaced by a space), so the only `<<` tokens that can appear
/// are quoted ones (`echo "<<EOF"`), which classify as ordinary words.
fn non_flag_path_args(segment: &str) -> Vec<String> {
    let words: Vec<&str> = segment.split_whitespace().collect();
    let Some(cmd_idx) = super::find_first_command_word_index(&words) else {
        return vec![];
    };

    let mut args = Vec::new();
    let mut skip_redirect_target = false;

    for w in &words[cmd_idx + 1..] {
        if skip_redirect_target {
            skip_redirect_target = false;
            continue;
        }

        if w.starts_with('-') {
            continue;
        }

        if let TokenKind::Redirect { needs_target } = classify_shell_token(w) {
            skip_redirect_target = needs_target;
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

/// Result of classifying a whitespace-split shell token.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TokenKind {
    /// Not a redirect or heredoc — pass through.
    Regular,
    /// A redirect operator.  When `needs_target`, the caller should skip
    /// the next whitespace-separated word (the redirect target).
    Redirect { needs_target: bool },
}

/// Classify a whitespace-split token as a shell redirect operator.
///
/// Returns [`TokenKind::Redirect`] with `needs_target` indicating whether
/// the operator expects a following word (e.g. `>` → needs target, `2>&1` →
/// self-contained), or [`TokenKind::Regular`] for anything else.
///
/// Heredoc markers (`<<EOF`, `<<-EOF`, `3<<EOF`) classify as self-contained
/// redirects (the marker embeds its delimiter). The read-only guard only
/// ever sees heredoc-stripped segments — [`strip_heredoc_bodies`] replaces
/// markers with spaces — so this is defensive; there is no heredoc body to
/// skip here.
///
/// # Ordering invariants
///
/// Exact-match checks for `>&` and `&>`/`&>>` MUST precede the pattern-based
/// checks because those patterns would also match `starts_with('>')` and
/// `contains("&>")` respectively, but with the wrong `needs_target` value.
///
/// 1. `>&` exact-match check precedes `starts_with('>')` — `>&` also starts
///    with `>` but has `needs_target: true` (standalone operator), while
///    combined forms like `>&2` fall through to `starts_with('>')` which
///    yields `needs_target: false`.
/// 2. `&>` / `&>>` exact-match checks precede `contains("&>")` — both
///    contain `&>` but have `needs_target: true` as standalone operators,
///    while combined forms like `&>/dev/null` fall through to
///    `contains("&>")` which yields `needs_target: false`.
///
/// # Design note
///
/// This is a **token-level** classifier used by [`non_flag_path_args`].  It is
/// NOT used by [`has_disallowed_redirect`], which operates at a different
/// abstraction level (character-based with quote-state awareness).  Those two
/// functions have distinct semantics and are deliberately kept separate.
fn classify_shell_token(w: &str) -> TokenKind {
    // ── Exact-match redirect operators ────────────────────────────
    // NOTE: `>&` and `&>`/`&>>` are checked via exact `match` BEFORE the
    // pattern-based checks below (see ordering invariants in doc comment).
    match w {
        // Operators that consume the next word as their target:
        ">" | ">&" | ">>" | ">|" | "<" | "<&" | "<>" | "&>" | "&>>" => {
            return TokenKind::Redirect { needs_target: true };
        }
        // Self-contained fd-merge — no target to skip:
        "2>&1" | "1>&2" => {
            return TokenKind::Redirect {
                needs_target: false,
            };
        }
        _ => {}
    }

    // ── Herestrings (`<<<` / `{digit}<<<`) ────────────────────────
    // A herestring has no body: it behaves like a redirect that consumes the
    // next word as its content. Classifying it as a heredoc would make the
    // caller skip everything after it — hiding a real redirect on the same
    // line (a live-reproduced guard bypass).
    if w.starts_with("<<<")
        || (w.len() > 3 && w.as_bytes()[0].is_ascii_digit() && w.ends_with("<<<"))
    {
        return TokenKind::Redirect { needs_target: true };
    }

    // ── Standalone digit-prefixed redirect ────────────────────────
    // e.g. 2>, 10>, 3< — expects a target word
    if is_digit_suffix_redirect(w, b'>') || is_digit_suffix_redirect(w, b'<') {
        return TokenKind::Redirect { needs_target: true };
    }

    // ── Combined redirect tokens (operator merged with target) ────
    // e.g. >/dev/null, >>file, </dev/null, <&2, <>file
    if w.starts_with('>') || w.starts_with('<') {
        return TokenKind::Redirect {
            needs_target: false,
        };
    }

    // ── Combined fd-prefixed redirect (digits + redirect) ─────────
    // e.g. 2>/dev/null, 3</dev/null, 1>/tmp/out
    if w.len() > 1 && w.as_bytes()[0].is_ascii_digit() && (w.contains('>') || w.contains('<')) {
        return TokenKind::Redirect {
            needs_target: false,
        };
    }

    // ── Combined bash &> redirect ─────────────────────────────────
    // e.g. &>/dev/null, &>>file
    if w.contains("&>") {
        return TokenKind::Redirect {
            needs_target: false,
        };
    }

    TokenKind::Regular
}

/// True when a path argument contains a `$` variable that failed to resolve —
/// used to tailor rejection messages with the recognized temp-variable
/// spellings (a literal temp path, or `NAME=$(mktemp -d)`).
fn has_unresolved_var_path(segment: &str, state: &ValidationState) -> bool {
    non_flag_path_args(segment)
        .iter()
        .any(|p| p.contains('$') && resolve_path_word(p, state).is_none())
}

/// True when every explicit path argument resolves under an allowed temp root.
fn scratch_paths_under_temp(segment: &str, state: &ValidationState) -> bool {
    let paths = non_flag_path_args(segment);
    !paths.is_empty()
        && paths.iter().all(|p| {
            resolve_path_word(p, state).is_some_and(|path| is_path_under_temp(&path, state.ctx))
        })
}

/// True when a temp mutator's path arguments all resolve under an allowed
/// temp root. `cp` is special-cased: only the destination must be under temp
/// (sources are read-only — copying from anywhere into temp is allowed;
/// copying into the workspace stays blocked, contract decision 8).
fn temp_mutator_paths_under_temp(segment: &str, first_word: &str, state: &ValidationState) -> bool {
    if first_word == "cp" {
        return cp_destination_under_temp(segment, state);
    }
    scratch_paths_under_temp(segment, state)
}

/// Identify the `cp` destination (contract decision 8): the value of
/// `-t`/`--target-directory`, or the last non-flag path argument. With no
/// identifiable destination the command is rejected.
fn cp_destination_under_temp(segment: &str, state: &ValidationState) -> bool {
    let words: Vec<&str> = segment.split_whitespace().collect();
    let Some(cmd_idx) = super::find_first_command_word_index(&words) else {
        return false;
    };
    let rest = &words[cmd_idx + 1..];

    for flag in ["-t", "--target-directory"] {
        if let Some(val) = flag_value(rest, flag) {
            return resolve_path_word(val, state)
                .is_some_and(|p| is_path_under_temp(&p, state.ctx));
        }
    }
    if let Some(val) = flag_value_equals(rest, "--target-directory=") {
        return resolve_path_word(val, state).is_some_and(|p| is_path_under_temp(&p, state.ctx));
    }

    match non_flag_path_args(segment).last() {
        Some(dest) => {
            resolve_path_word(dest, state).is_some_and(|p| is_path_under_temp(&p, state.ctx))
        }
        None => false, // no identifiable destination → reject
    }
}

/// Check a single command segment for unsafe operations.
///
/// `state` carries the tracked current directory and temp-variable bindings
/// across segments (so `cd /tmp && touch f` resolves `f` against `/tmp`).
fn check_segment(segment: &str, state: &mut ValidationState) -> Result<(), String> {
    let trimmed = segment.trim();
    if trimmed.is_empty() {
        return Ok(());
    }

    // ── CWD tracking (contract decision 7) ─────────────────────────
    // A literal absolute `cd` into a directory that exists at validation time
    // (or was created by `mkdir` earlier in the chain) updates the tracked
    // CWD, after skipping `cd` option flags (`-P`/`-L`/`--`); a flag-only
    // form (`cd -P` → `$HOME`) or invalid option resets. All other cd forms —
    // relative `cd` whose target does not provably resolve to an existing or
    // created temp dir, `cd ..`, `cd -`, bare `cd`, `cd $VAR`, `cd ~`,
    // pushd/popd — reset tracking to fail-closed: subsequent relative path
    // arguments and redirect targets reject.
    //
    // The verb is located after env assignments and the shell prefixes that
    // forward their command to the current shell (`command`/`builtin`/`time`/
    // `eval` — `command cd /etc` moves the real CWD). Non-forwarding prefixes
    // (`sudo`/`env`/`nice`/…) run the `cd` in a child shell, so the real CWD
    // never changes and the segment is not routed here.
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let first_effective = words
        .iter()
        .find(|w| {
            !super::is_env_assignment(w) && !matches!(**w, "command" | "builtin" | "time" | "eval")
        })
        .copied()
        .unwrap_or("");
    if matches!(first_effective, "cd" | "pushd" | "popd") {
        update_cwd_for_cd(trimmed, state);
        return Ok(());
    }

    // ── Temp-variable bindings (contract decision 5) ───────────────
    // `export X=...`, plain `X=...` segments, and env-prefix forms
    // (`TMPDIR=/tmp cmd`) bind the temp variables; a non-temp binding poisons
    // the variable, a temp-root binding clears the poison.
    apply_env_bindings(trimmed, state);

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

    // Scratch mutators (tee, touch, mkdir) are allowed when all explicit
    // path arguments are under an allowed temp directory.
    if SCRATCH_MUTATORS.contains(&first_word) {
        if scratch_paths_under_temp(trimmed, state) {
            // `mkdir` creates the directories it names — record those whose
            // creation is provable (all `-p` targets, and non-`-p` targets
            // whose parent already exists) so a later `cd` in the same chain
            // can track them even though they did not exist at validation
            // time. A non-`-p` target under a missing parent errors at
            // runtime and must not be treated as created (fail-closed).
            if first_word == "mkdir" {
                record_mkdir_targets(trimmed, state);
            }
            return Ok(());
        }
        return reject(
            trimmed,
            &format!(
                "`{first_word}` is not allowed outside temp directories — it modifies the workspace."
            ),
            if has_unresolved_var_path(trimmed, state) {
                "use a literal path under /tmp, or bind the directory first with `NAME=$(mktemp -d)` and reference `$NAME`."
            } else {
                "use read-only alternatives to inspect files, e.g. `cat`, `head`, `tail`, `ls`, `file`, `stat`."
            },
        );
    }

    // Temp mutators (cp, mv, rm, gzip, etc.) are allowed when their path
    // arguments target temp directories. The prompt tells agents that
    // writing to the OS temp directory is allowed.
    if TEMP_MUTATORS.contains(&first_word) {
        if temp_mutator_paths_under_temp(trimmed, first_word, state) {
            return Ok(());
        }
        return reject(
            trimmed,
            &format!(
                "`{first_word}` is not allowed outside temp directories — it modifies files outside /tmp."
            ),
            if has_unresolved_var_path(trimmed, state) {
                "use a literal path under /tmp, or bind the directory first with `NAME=$(mktemp -d)` and reference `$NAME`."
            } else {
                "use paths under /tmp, /var/tmp, or the OS temp directory, or use read-only alternatives like `cat`, `head`, `tail`, `ls`, `file`, `stat`."
            },
        );
    }

    // Unconditional rejections — commands that always mutate.
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
        if first_word == check.verb && (check.predicate)(trimmed, state) {
            return reject(trimmed, check.rejection, check.suggestion);
        }
    }

    Ok(())
}

/// Update the tracked CWD for a `cd`/`pushd`/`popd` segment (decision 7).
///
/// A literal absolute `cd` into a temp-root path tracks only when the
/// directory exists at validation time or was provably created by `mkdir`
/// earlier in the same command chain: a nonexistent target fails the `cd` at
/// runtime, leaving the real CWD in place, so tracking it would approve a
/// chained write that actually lands elsewhere (or escapes a `..` chain past
/// the temp root when the tracked depth does not match). A literal absolute
/// `cd` elsewhere tracks only when the directory exists. Every other form —
/// relative `cd` whose target does not provably resolve to an existing or
/// created temp dir, `cd ..`, `cd -`, bare `cd`, `cd $VAR`, `cd ~`,
/// pushd/popd — resets tracking to fail-closed (`None` → relative paths
/// reject). `cd` option flags (`-P`/`-L`/`--`) are skipped before the target
/// is resolved; a flag-only form (`cd -P`) targets `$HOME` and an invalid
/// option (`-e`, `-@`, …) errors at runtime, so both reset fail-closed.
/// Relative targets are expanded against the tracked state before tracking,
/// so `$VAR`, `~`, `-` and `$OLDPWD` cannot smuggle a non-temp location past
/// the guard.
fn update_cwd_for_cd(segment: &str, state: &mut ValidationState) {
    let words: Vec<&str> = segment.split_whitespace().collect();
    let Some(cd_idx) = words
        .iter()
        .position(|w| matches!(*w, "cd" | "pushd" | "popd"))
    else {
        return;
    };
    // pushd/popd always reset fail-closed; only `cd` tracks.
    if words[cd_idx] != "cd" {
        state.cwd = None;
        return;
    }
    // Skip option words before resolving the real target. `-P`/`-L` (and
    // combined forms) are valid; `--` ends option parsing; any other `-…`
    // option (`-e`, `-@`, `-x`, …) errors at runtime, so the `cd` never
    // happens and the real CWD stays put — tracking would approve a chained
    // write that lands there. A bare `cd`/`cd -P` (no target after the
    // flags) targets `$HOME`. All of these reset fail-closed; `-` alone is
    // the `$OLDPWD` target, never an option.
    let mut i = cd_idx + 1;
    let mut options_ended = false;
    let target = loop {
        let Some(w) = words.get(i) else { break None };
        if !options_ended && w.starts_with('-') && *w != "-" {
            if *w == "--" {
                options_ended = true;
            } else if !w[1..].bytes().all(|b| matches!(b, b'P' | b'L')) {
                break None;
            }
            i += 1;
            continue;
        }
        break Some(*w);
    };
    let Some(target) = target else {
        state.cwd = None;
        return;
    };
    if target.starts_with('/') {
        let p = std::path::PathBuf::from(target);
        if is_path_under_temp(&p, state.ctx) {
            // Track only when the directory exists or was provably created by
            // `mkdir` earlier in this chain. A nonexistent target — or one
            // whose raw path has a missing prefix component (`/tmp/a/../b`
            // with `a` absent) — fails the `cd` at runtime and a chained
            // write would land in the real CWD (fail-closed).
            let normalized = crate::tools::path::normalize_path(&p);
            state.cwd = if path_exists_or_created(&p, &normalized, state) {
                Some(p)
            } else {
                None
            };
        } else {
            // Filesystem existence checks are expected and acceptable
            // (decision 7); a nonexistent target fails the command, so
            // tracking resets fail-closed.
            state.cwd = if p.is_dir() { Some(p) } else { None };
        }
    } else {
        // Relative cd under a tracked temp cwd tracks iff the target — after
        // variable expansion — resolves to a path under a temp root that
        // exists at validation time or was created by `mkdir` earlier in the
        // chain. A nonexistent target fails the `cd` at runtime, leaving the
        // real CWD one level shallower: tracking the deeper path would let a
        // `..` chained write climb past the temp root (fail-closed). `~` and
        // `-` never resolve under temp, and unresolvable targets (`$HOME`,
        // `$OLDPWD`, unbound variables) reject.
        let Some(cwd) = &state.cwd else {
            state.cwd = None;
            return;
        };
        if !is_path_under_temp(cwd, state.ctx) {
            state.cwd = None;
            return;
        }
        if target.starts_with('~') || target == "-" {
            state.cwd = None;
            return;
        }
        let Some(resolved) = resolve_path_word(target, state) else {
            state.cwd = None;
            return;
        };
        let normalized = crate::tools::path::normalize_path(&resolved);
        state.cwd = if is_path_under_temp(&normalized, state.ctx)
            && path_exists_or_created(&resolved, &normalized, state)
        {
            Some(normalized)
        } else {
            None
        };
    }
}

/// A `cd`/`mktemp` target is provably reachable at runtime when the raw path
/// resolves through the filesystem (`is_dir` follows every component,
/// matching `chdir`/`stat` semantics) or — for `..`-free paths — was recorded
/// by an earlier `mkdir` in the chain. A raw path containing `..` is only
/// provable when it already exists: a missing prefix component fails the
/// command at runtime even when the normalized tail exists.
fn path_exists_or_created(
    raw: &std::path::Path,
    normalized: &std::path::Path,
    state: &ValidationState,
) -> bool {
    raw.is_dir()
        || (!raw
            .components()
            .any(|c| c == std::path::Component::ParentDir)
            && state.created_dirs.contains(normalized))
}

/// Record temp directories created by an approved `mkdir` segment (normalized
/// paths) so a later `cd` in the same command chain can track them even though
/// they did not exist at validation time. With `-p`/`--parents` the ancestors
/// are recorded too — `mkdir -p` creates them along the way.
fn record_mkdir_targets(segment: &str, state: &mut ValidationState) {
    let parents = segment
        .split_whitespace()
        .any(|w| w == "-p" || w == "--parents");
    for arg in non_flag_path_args(segment) {
        let Some(resolved) = resolve_path_word(&arg, state) else {
            continue;
        };
        let dir = crate::tools::path::normalize_path(&resolved);
        if !is_path_under_temp(&dir, state.ctx) {
            continue;
        }
        if parents {
            // `-p` creates the whole chain, so every ancestor is provable.
            state.created_dirs.insert(dir.clone());
            let mut d = dir;
            while let Some(parent) = d.parent() {
                d = parent.to_path_buf();
                if !is_path_under_temp(&d, state.ctx) {
                    break;
                }
                state.created_dirs.insert(d.clone());
            }
        } else if dir.parent().is_some_and(|p| {
            // Without `-p`, mkdir fails when the parent is absent at runtime:
            // a target is only provably created when its immediate parent
            // exists at validation time or was created by `mkdir` earlier in
            // this chain.
            p.is_dir()
                || state
                    .created_dirs
                    .contains(&crate::tools::path::normalize_path(p))
        }) {
            state.created_dirs.insert(dir);
        }
    }
}

/// Apply temp-variable bindings from a segment (decision 5): `export X=...`,
/// plain `X=...` assignment segments, and leading env-prefix assignments
/// (`TMPDIR=/tmp cmd`). Bare `export` / `export NAME` are no-ops.
fn apply_env_bindings(segment: &str, state: &mut ValidationState) {
    let words = split_words_keeping_substitutions(segment);
    if words.first() == Some(&"export") {
        for w in &words[1..] {
            if let Some((name, value)) = w.split_once('=') {
                apply_single_binding(name, value, state);
            }
            // `export NAME` without a value neither clears poisoning nor
            // enables anything new (decision 5).
        }
        return;
    }
    for w in words.iter().take_while(|w| super::is_env_assignment(w)) {
        if let Some((name, value)) = w.split_once('=') {
            apply_single_binding(name, value, state);
        }
    }
}

/// Split a segment into whitespace-separated words, keeping `$(...)` /
/// backtick substitutions whole — their bodies may contain spaces, and
/// `NAME=$(mktemp -d)` must stay one assignment word.
fn split_words_keeping_substitutions(s: &str) -> Vec<&str> {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    while i < s.len() {
        let c = s[i..].chars().next().expect("i < len");
        // Substitution starts are recognized inside double quotes too, but are
        // literal inside single quotes or after an escape backslash.
        if !escaped && !in_single && (c == '$' && bytes.get(i + 1) == Some(&b'(') || c == '`') {
            let next = if c == '$' {
                let (_, next) = find_substitution_end(s, i + 2);
                next
            } else {
                let (_, next) = find_backtick_end(s, i + 1);
                next
            };
            i = next;
            continue;
        }
        if !super::track_char_context(c, &mut in_single, &mut in_double, &mut escaped) {
            i += c.len_utf8();
            continue;
        }
        if c.is_whitespace() {
            if start < i {
                out.push(&s[start..i]);
            }
            i += c.len_utf8();
            start = i;
            continue;
        }
        i += c.len_utf8();
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// Bind a single `NAME=value` assignment to a tracked variable.
///
/// Any variable assigned a value that provably resolves under a temp root is
/// tracked for path resolution (TMPDIR/TMP/TEMP as before, plus e.g.
/// `SNAP=/tmp/x`). `NAME=$(mktemp -d)` / `` NAME=`mktemp -d` `` bind the
/// variable to a fresh temp root with an unknown concrete value. A non-temp
/// binding poisons the variable (fail-closed); a temp binding clears the
/// poison.
///
/// The stored value is the *resolved* path (shell expansion happens at
/// assignment time, so `export TMPDIR="$TMPDIR/x"` binds the expanded path).
fn apply_single_binding(name: &str, value: &str, state: &mut ValidationState) {
    // Strip balanced surrounding quotes from the value (`export TMPDIR="/tmp"`).
    // A single-quoted value is a literal — no expansion, no substitution —
    // so it can never be an mktemp temp binding.
    let (clean, single_quoted) = strip_outer_quotes(value).map_or((value, false), |(c, q)| (c, q));
    // `NAME=$(mktemp -d)` binds a fresh temp root with an unknown value.
    if !single_quoted {
        if let Some(binding) = mktemp_binding(clean, state) {
            state.vars.insert(name.to_string(), binding);
            return;
        }
        // Any other command substitution has an unprovable output. Resolving
        // the raw `$(...)` string as a literal path would fabricate a temp
        // value against a tracked temp cwd (`cd /tmp; SNAP=$(mktemp -d -p
        // /tmp/nonexistent_x)`): at runtime mktemp errors, SNAP binds empty,
        // and the chained write lands outside every temp root. Bind such
        // variables poisoned (fail-closed).
        if substitution_body(clean).is_some() {
            state.vars.insert(
                name.to_string(),
                VarBinding {
                    value: Some(clean.to_string()),
                    poisoned: true,
                },
            );
            return;
        }
    }
    let resolved = if !clean.is_empty() && !clean.starts_with('~') {
        if single_quoted && clean.contains('$') {
            None // literal `$` (e.g. `'$(mktemp -d)'`) never resolves under temp
        } else {
            resolve_path_word(clean, state)
        }
    } else {
        None
    };
    let under_temp = resolved
        .as_ref()
        .is_some_and(|p| is_path_under_temp(p, state.ctx));
    state.vars.insert(
        name.to_string(),
        VarBinding {
            value: Some(
                resolved.map_or_else(|| clean.to_string(), |p| p.to_string_lossy().into_owned()),
            ),
            poisoned: !under_temp,
        },
    );
}

/// Extract the command body from a `$(...)` or `` `...` `` substitution
/// (outer quotes already stripped).
fn substitution_body(value: &str) -> Option<&str> {
    if let Some(inner) = value.strip_prefix("$(") {
        return inner.strip_suffix(')');
    }
    if let Some(inner) = value.strip_prefix('`') {
        return inner.strip_suffix('`');
    }
    None
}

/// Detect a `NAME=$(mktemp ...)` / `` NAME=`mktemp ...` `` assignment and
/// return a temp-tagged binding (unknown concrete value, provably under a
/// temp root). Returns `None` when the value is not an mktemp substitution or
/// when the invocation's landing directory is not provably under a temp root
/// (the variable then falls through to the normal binding path, which poisons
/// it — fail-closed).
///
/// The landing directory comes from `-p DIR` / `--tmpdir=DIR` or from a
/// positional template. `-p DIR` is honored only when DIR provably exists (at
/// validation time or created by `mkdir` earlier in the chain): mktemp never
/// creates that directory, so a missing one makes it error and bind the
/// variable empty — a false temp anchor. The template is honored only when it
/// provably resolves under a temp root (an absolute temp path, or a variable
/// expanding to one). Absolute non-temp templates (e.g. under /etc, the
/// workspace, $HOME), relative templates (resolved against the shell cwd) and
/// the `-- <template>` forms carrying them all fail closed. The
/// space-separated `--tmpdir DIR` form is not provable — GNU treats the value
/// as optional (needs `=`) and errors at runtime there, while macOS joins
/// `DIR` as a template under /tmp instead of using it as the target dir — so
/// it fails closed; only the `=` form binds. An unrecognized option (e.g.
/// `--suffix=`, which macOS mktemp rejects) or a second template makes mktemp
/// error at runtime — the variable would bind empty, so a temp anchor would
/// be a false one; fail closed.
fn mktemp_binding(value: &str, state: &ValidationState) -> Option<VarBinding> {
    let inner = substitution_body(value)?;
    let mut words = inner.split_whitespace();
    if words.next()? != "mktemp" {
        return None;
    }
    let args: Vec<&str> = words.collect();
    let mut target_dir: Option<&str> = None;
    let mut template: Option<&str> = None;
    let mut after_ddash = false;
    let mut i = 0;
    while i < args.len() {
        let a = args[i];
        if a == "--" {
            after_ddash = true;
            i += 1;
            continue;
        }
        if !after_ddash {
            // `-p DIR` / `--tmpdir=DIR` override the target directory; the
            // result is only temp when that directory provably exists under
            // temp (see the gating below the parse loop).
            if a == "-p" {
                if i + 1 < args.len() {
                    target_dir = Some(args[i + 1]);
                    i += 2;
                    continue;
                }
                return None; // missing dir value → fail-closed
            }
            // The space-separated `--tmpdir DIR` form is not provable across
            // platforms (see the doc comment) → fail-closed; only `=` binds.
            if a == "--tmpdir" {
                return None;
            }
            if let Some(dir) = a.strip_prefix("--tmpdir=") {
                target_dir = Some(dir);
                i += 1;
                continue;
            }
            // `-t prefix` takes a value (BSD spelling); the result lands under
            // the OS temp root (`temp_dir()`), an allowed temp root.
            if a == "-t" {
                if i + 1 < args.len() {
                    i += 2;
                    continue;
                }
                return None; // missing prefix value → mktemp errors at runtime
            }
            // Known boolean flags do not affect where the result lands.
            if matches!(a, "-d" | "-q" | "-u" | "--dry-run" | "--quiet") {
                i += 1;
                continue;
            }
            if a.starts_with('-') {
                return None; // unrecognized option (e.g. `--suffix=`, which
                // macOS mktemp rejects) → fail-closed
            }
        }
        // First positional argument is the template; a second one makes
        // mktemp error at runtime → fail-closed.
        if template.is_some() {
            return None;
        }
        template = Some(a);
        i += 1;
    }
    // The template determines the landing path when present: it must
    // provably resolve under a temp root. Relative templates resolve against
    // the tracked CWD — the shell cwd at runtime — so they fail closed unless
    // that cwd is already a tracked temp root. A template without trailing
    // X's makes mktemp error at runtime and bind the variable empty, so no
    // temp anchor may be created (fail-closed). mktemp creates the final
    // component, so the raw parent must be traversable at runtime — a missing
    // prefix component (`a/../b.XXXXXX` with `a` absent) errors mktemp even
    // when the normalized tail exists.
    if let Some(t) = template {
        let clean = strip_outer_quotes(t).map_or(t, |(c, _)| c);
        if !clean.ends_with("XXXXXX") {
            return None;
        }
        let p = resolve_path_word(clean, state)?;
        let parent = p.parent()?;
        let parent_norm = crate::tools::path::normalize_path(parent);
        if !is_path_under_temp(&p, state.ctx)
            || !path_exists_or_created(parent, &parent_norm, state)
        {
            return None;
        }
    }
    match target_dir {
        None => Some(VarBinding {
            value: None,
            poisoned: false,
        }),
        Some(dir) => {
            let clean = strip_outer_quotes(dir).map_or(dir, |(c, _)| c);
            let raw = resolve_path_word(clean, state)?;
            let normalized = crate::tools::path::normalize_path(&raw);
            // mktemp never creates the `-p` directory: it errors when the dir
            // does not exist (or its raw path is un-traversable, e.g. a
            // missing prefix component) and the variable binds empty (false
            // temp anchor). Track only when the dir provably exists.
            if is_path_under_temp(&normalized, state.ctx)
                && path_exists_or_created(&raw, &normalized, state)
            {
                Some(VarBinding {
                    value: None,
                    poisoned: false,
                })
            } else {
                None
            }
        }
    }
}

// ── Git-specific checks ──────────────────────────────────────────────────

/// Mutation flags/verbs for `git branch` (any of these makes the command mutating).
///
/// NOTE: `-f`/`--force` bypasses safety checks (force-create / force-delete).
///       `-u`/`--set-upstream-to` sets upstream tracking (requires force with `-f`).
///       `--set-upstream-to=value` notation is also caught via prefix matching.
///       `--track`/`--no-track` create tracking branches or override config.
const GIT_BRANCH_MUTATIONS: &[&str] = &[
    "-d",
    "-D",
    "-m",
    "-M",
    "-c",
    "-C",
    "-f",
    "--force",
    "-u",
    "--set-upstream-to",
    "--track",
    "--no-track",
    "--delete",
    "--move",
    "--copy",
    "--edit-description",
];

/// Mutation flags for `git tag` (any of these makes the command mutating).
///
/// NOTE: `-f`/`--force` bypasses safety checks (force-create / force-delete / force-replace).
///       `-m`/`--message`, `-F`/`--file`, and `-e`/`--edit` attach or edit tag messages.
const GIT_TAG_MUTATIONS: &[&str] = &[
    "-d",
    "--delete",
    "-a",
    "-s",
    "-u",
    "-f",
    "--force",
    "-m",
    "--message",
    "-F",
    "--file",
    "-e",
    "--edit",
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

/// True when the command has a git dry-run token: `-n`, `--dry-run`, or a
/// combined short flag containing `n` (e.g. `clean -ndx`). Long flags are
/// excluded so `--name-only`-style tokens don't count.
fn has_dry_run_token(command: &str) -> bool {
    command.split_whitespace().any(|w| {
        w == "--dry-run"
            || (w.starts_with('-') && !w.starts_with("--") && w.len() > 1 && w[1..].contains('n'))
    })
}

/// True when the command has a git force token: `-f`, `--force`, or a
/// combined short flag containing `f` (e.g. `clean -fd`, `push -fn`).
/// Any force token blocks dry-run clean/push (decision 3).
fn has_force_token(command: &str) -> bool {
    command.split_whitespace().any(|w| {
        w == "--force"
            || (w.starts_with('-') && !w.starts_with("--") && w.len() > 1 && w[1..].contains('f'))
    })
}

/// Phase-3 read-only git allowlist rules (contract decisions 3/4 and the
/// phase-3 scope): `stash show`, `config` read forms, `rebase --show-current`,
/// `push --dry-run`/`-n`, `clean -n`/`--dry-run`, and `submodule status`.
///
/// Returns `Some(result)` when `subcommand` matches one of these prefixes
/// (the caller returns it directly), `None` when the subcommand is outside
/// these rules and should fall through to the general safe-list match.
///
/// Matching is exact: mutating siblings stay blocked (`stash pop`,
/// `config user.name Egor`, `rebase --continue`, `push origin main`,
/// `clean -f`, `submodule foreach`), and any force token blocks the dry-run
/// clean/push forms.
fn check_git_read_only_extensions(trimmed: &str, subcommand: &str) -> Option<Result<(), String>> {
    // git stash: `list` and `show` are read-only inspection; everything else
    // (push, pop, apply, drop, ...) modifies the working tree. Matching is
    // exact at the stash subcommand word — a mutation like
    // `git stash push -m "stash show"` stays blocked even though its message
    // contains a read-form word.
    if subcommand.starts_with("stash") {
        let words: Vec<&str> = subcommand.split_whitespace().collect();
        let stash_cmd = words.get(1).copied().unwrap_or("");
        if matches!(stash_cmd, "show" | "list") {
            return Some(Ok(()));
        }
        return Some(reject(
            trimmed,
            "`git stash` is not allowed — it modifies the working tree.",
            "use `git stash list` to view stashes, or `git diff` to preview changes.",
        ));
    }

    // git config read/write rule (decision 4): exactly one positional after
    // `config` (key read) is allowed; two positionals (key + value) write.
    // Explicit get forms (--list, -l, --get, --get-all, --get-regexp,
    // --name-only) are allowed; write/edit forms are blocked.
    if subcommand.starts_with("config") {
        let words: Vec<&str> = subcommand.split_whitespace().collect();
        let rest = &words[1..];
        if rest.iter().any(|w| {
            matches!(
                *w,
                "--list" | "-l" | "--get" | "--get-all" | "--get-regexp" | "--name-only"
            )
        }) {
            return Some(Ok(()));
        }
        if rest.iter().any(|w| {
            matches!(
                *w,
                "--add"
                    | "--unset"
                    | "--unset-all"
                    | "--edit"
                    | "--remove-section"
                    | "--rename-section"
                    | "--replace-all"
            )
        }) {
            return Some(reject(
                trimmed,
                "`git config` write/edit forms are not allowed — they modify repository or global config.",
                "use `git config user.name` (key read), `git config --list`, or `git config --get <key>` to inspect configuration.",
            ));
        }
        match rest.iter().filter(|w| !w.starts_with('-')).count() {
            0 | 1 => Some(Ok(())), // bare `git config` / key read
            _ => Some(reject(
                trimmed,
                "`git config` with a value is not allowed — it writes configuration.",
                "use `git config user.name` (key read) or `git config --list` to inspect configuration.",
            )),
        }
    } else if subcommand.starts_with("rebase") {
        // git rebase --show-current is a pure read; every other rebase form
        // mutates the branch history.
        let words: Vec<&str> = subcommand.split_whitespace().collect();
        if words.len() >= 2 && words[1] == "--show-current" {
            return Some(Ok(()));
        }
        Some(reject(
            trimmed,
            "`git rebase` is not allowed — it rewrites branch history.",
            "use `git rebase --show-current` to see the in-progress rebase, or `git log` to inspect history.",
        ))
    } else if subcommand.starts_with("push") {
        // git push --dry-run (-n/--dry-run) performs a network read with no
        // local mutation; any force token in the same command blocks it
        // (decision 3).
        if has_dry_run_token(subcommand) && !has_force_token(subcommand) {
            Some(Ok(()))
        } else {
            Some(reject(
                trimmed,
                "`git push` is not allowed — it writes to a remote repository.",
                "use `git push --dry-run` to preview what would be pushed without sending anything.",
            ))
        }
    } else if subcommand.starts_with("clean") {
        // git clean -n/--dry-run previews removals; any force token blocks it
        // (decision 3).
        if has_dry_run_token(subcommand) && !has_force_token(subcommand) {
            Some(Ok(()))
        } else {
            Some(reject(
                trimmed,
                "`git clean` is not allowed — it deletes untracked files.",
                "use `git clean -n` to preview what would be removed without deleting anything.",
            ))
        }
    } else if subcommand.starts_with("submodule") {
        // git submodule: `status` (and bare `submodule`, which prints a status
        // summary) are read-only; foreach/update/add/... mutate submodules.
        let words: Vec<&str> = subcommand.split_whitespace().collect();
        match words.get(1) {
            None | Some(&"status") => Some(Ok(())),
            Some(sub) => Some(reject(
                trimmed,
                &format!("`git submodule {sub}` is not allowed — it modifies submodules."),
                "use `git submodule status` to inspect submodule state.",
            )),
        }
    } else {
        None
    }
}

fn check_git_segment(segment: &str) -> Result<(), String> {
    let trimmed = segment.trim();

    // Extract the git subcommand by skipping "git" and global flags
    let subcommand = extract_git_subcommand(trimmed);

    if subcommand.is_empty() || subcommand == "git" {
        return Ok(());
    }

    // Phase-3 read-only allowlist rules (stash/config/rebase/push/clean/
    // submodule). These run BEFORE the general safe-list match because the
    // subcommands are not (all) in GIT_SAFE_SUBCOMMANDS, and the stash check
    // must exempt `stash show` at this layer.
    if let Some(result) = check_git_read_only_extensions(trimmed, &subcommand) {
        return result;
    }

    // git mktag always writes a tag object to the object database — no read-only mode.
    if subcommand.starts_with("mktag") {
        return reject(
            trimmed,
            "`git mktag` is not allowed — it always writes a tag object to the object database.",
            "use `git verify-tag` or `git cat-file` to inspect existing tag objects.",
        );
    }

    // git mktree always writes a tree object to the object database — no read-only mode.
    if subcommand.starts_with("mktree") {
        return reject(
            trimmed,
            "`git mktree` is not allowed — it always writes a tree object to the object database.",
            "use `git ls-tree` to inspect existing tree objects.",
        );
    }

    // git merge-file without -p/--stdout overwrites the <current> file in-place.
    // Even with -p/--stdout, the --object-id variant writes to the object store.
    // Rather than a complex multi-flag check, reject entirely for safety.
    if subcommand.starts_with("merge-file") {
        return reject(
            trimmed,
            "`git merge-file` is not allowed — it mutates files or writes to the object database.",
            "use `git diff` to compare files, or `diff`/`diff3` for three-way comparisons.",
        );
    }

    // git merge-tree since Git 2.40 defaults to --write-tree, which creates tree
    // objects in the object database. The alternative --trivial-merge is deprecated.
    if subcommand.starts_with("merge-tree") {
        return reject(
            trimmed,
            "`git merge-tree` is not allowed — it writes tree objects in its default mode.",
            "use `git merge-base` to find the merge base, or `git diff-tree` to inspect trees.",
        );
    }

    let matched_safe = GIT_SAFE_SUBCOMMANDS
        .iter()
        .copied()
        .find(|safe| subcommand == *safe || subcommand.starts_with(&format!("{safe} ")));

    if matched_safe.is_none() {
        return Err(format!(
            "⚠️ Read-only mode: the `git {subcommand}` subcommand is not allowed — it may mutate the repository.\n\
             Command: `{trimmed}`\n\
             Allowed git subcommands for read-only mode: status, log, diff, show, blame, branch, tag, remote,\n\
             stash list/show, show-ref, ls-remote, submodule status, config reads, rebase --show-current,\n\
             push --dry-run, clean -n, and other inspection-only commands. Suggestion: use these for repository exploration."
        ));
    }

    // Additional mutation-flag checks for branch/tag/remote/hash-object/reflog
    match matched_safe {
        Some("branch") => {
            check_git_subcommand_mutation(&subcommand, "branch", GIT_BRANCH_MUTATIONS)?;
        }
        Some("tag") => check_git_subcommand_mutation(&subcommand, "tag", GIT_TAG_MUTATIONS)?,
        Some("remote") => {
            check_git_subcommand_mutation(&subcommand, "remote", GIT_REMOTE_MUTATIONS)?;
        }
        Some("hash-object") => {
            // git hash-object -w writes the object to the database; without -w
            // it only computes and outputs the hash (read-only). The -w flag can
            // appear anywhere in the argument list, so we search the full command.
            if has_flag(trimmed, "w") {
                return reject(
                    trimmed,
                    "`git hash-object -w` is not allowed — it writes objects to the object database.",
                    "use `git hash-object` without `-w` to compute the hash without storing the object.",
                );
            }
        }
        Some("reflog") => {
            // git reflog expire/delete mutate the reflog. Other subcommands
            // (show, list, exists, or bare `git reflog`) are read-only.
            let words: Vec<&str> = subcommand.split_whitespace().collect();
            if let Some(reflog_sub) = words.get(1)
                && (*reflog_sub == "expire" || *reflog_sub == "delete")
            {
                return reject(
                    trimmed,
                    &format!("`git reflog {reflog_sub}` is not allowed — it modifies the reflog."),
                    "use `git reflog show` or bare `git reflog` to view reflog entries.",
                );
            }
        }
        _ => {}
    }

    Ok(())
}

/// For a matched git subcommand, check for mutation flags/verbs across all
/// argument positions.
///
/// **Flag-based tokens** (those starting with `-`) are checked in all argument
/// positions — a flag like `-d` or `--delete` can never appear as a legitimate
/// branch or tag name.
///
/// **Bare-word tokens** (remote verbs like `add`, `remove`, `prune`) are checked
/// only at the first non-flag argument position. They cannot be safely checked
/// in all positions because they can collide with legitimate names — e.g.,
/// `git remote show add` is a valid read-only command where `add` is a remote
/// name, not a mutation verb.
///
/// For `branch` and `tag`: also rejects any bare (non-flag) first argument
/// (e.g., `git branch my-feature` or `git tag v1.0`), since these create
/// branches/tags rather than listing them.
///
/// Handles both exact flag matches and `flag=value` notation (e.g.,
/// `--set-upstream-to=origin/main`).
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

    // ── Name-creation check (branch/tag only) ──
    // Check if the first argument creates a new branch/tag by name.
    // This check stays on the first argument only — a bare name like
    // "feature" always creates, regardless of later positions.
    if let Some(first_arg) = words.get(1) {
        let is_name_creation = (subcommand_name == "branch" || subcommand_name == "tag")
            && !first_arg.starts_with('-');
        if is_name_creation {
            return Err(format!(
                "⚠️ Read-only mode: `git {subcommand}` is not allowed — it would create a {subcommand_name}.\n\
                 Suggestion: use `git {subcommand_name} --list` or `git {subcommand_name} --merged` to list existing ones."
            ));
        }
    }

    // Partition mutation tokens into flag-based and bare-word.
    let (flag_tokens, bare_tokens): (Vec<&str>, Vec<&str>) = mutation_tokens
        .iter()
        .copied()
        .partition(|t| t.starts_with('-'));

    // ── Flag-based mutation token check (all positions) ──
    // Tokens starting with `-` are safe to check in every argument position
    // because flags like `-d` or `--delete` can never be legitimate names.
    for arg in words.iter().skip(1) {
        if !arg.starts_with('-') {
            continue;
        }
        let is_mutating = flag_tokens.contains(arg)
            || flag_tokens
                .iter()
                .any(|t| arg.starts_with(&format!("{t}=")));
        if is_mutating {
            return Err(format!(
                "⚠️ Read-only mode: `git {subcommand}` is not allowed — it mutates.\n\
                 Suggestion: use `git {subcommand_name}` without mutation flags to list/inspect."
            ));
        }
    }

    // ── Bare-word mutation token check (first non-flag argument) ──
    // Bare-word tokens cannot be safely checked in all positions because
    // they can collide with legitimate names (e.g., `git remote show add`
    // where "add" is a remote name, not a mutation verb). Instead, skip
    // any leading flags and check the first non-flag argument.
    if !bare_tokens.is_empty()
        && let Some(first_non_flag_arg) = words.iter().skip(1).find(|w| !w.starts_with('-'))
    {
        let is_mutating = bare_tokens.contains(first_non_flag_arg)
            || bare_tokens
                .iter()
                .any(|t| first_non_flag_arg.starts_with(&format!("{t}=")));
        if is_mutating {
            return Err(format!(
                "⚠️ Read-only mode: `git {subcommand}` is not allowed — it mutates.\n\
                     Suggestion: use `git {subcommand_name}` without mutation flags to list/inspect."
            ));
        }
    }

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

    // ── Help/version exemption (decision 1) ────────────────────────
    // `-h`/`--help`/`-V`/`--version` appearing as a standalone token BEFORE a
    // `--` separator is a pure read and is allowed for ANY cargo subcommand,
    // including `run`. Tokens after `--` stay blocked (`cargo run -- --help`).
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let help_version_before_ddash = words
        .iter()
        .take_while(|w| **w != "--")
        .any(|w| matches!(*w, "-h" | "--help" | "-V" | "--version"));
    if help_version_before_ddash {
        return Ok(());
    }

    // ── Specific rejection messages for subcommands that modify source files ──
    // These are checked before the allowlist so they get tailored suggestions
    // instead of the generic "use cargo check, cargo test, ..." message.
    match base {
        "update" => {
            if words.contains(&"--dry-run") {
                // Dry-run previews changes without modifying Cargo.lock
                // (phase 3, decision 1/10).
                return Ok(());
            }
            return reject(
                trimmed,
                "`cargo update` is not allowed — it modifies Cargo.lock.",
                "switch to full shell mode to use `cargo update` \
                 (or `cargo update --dry-run` to preview changes without modifying Cargo.lock).",
            );
        }
        "generate-lockfile" => {
            return reject(
                trimmed,
                "`cargo generate-lockfile` is not allowed — it creates or overwrites Cargo.lock.",
                "switch to full shell mode to use `cargo generate-lockfile`.",
            );
        }
        "run" => {
            return reject(
                trimmed,
                "`cargo run` is not allowed — it executes the built binary, which may write files.",
                "build with `cargo build` first, then invoke the built binary directly, e.g. `./target/debug/<bin> --help`.",
            );
        }
        _ => {}
    }

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

/// Return the value of a flag that takes a separate word as its argument.
/// e.g., for `curl -o /tmp/file URL`, calling `flag_value(parts, "-o")` returns `Some("/tmp/file")`.
/// Returns `None` when the flag is not found, is the last token, or its value starts with `-`.
fn flag_value<'a>(parts: &'a [&'a str], flag: &str) -> Option<&'a str> {
    parts.windows(2).find_map(|w| {
        if w[0] == flag {
            let val = w[1];
            // If the value starts with `-`, it's likely another flag, not a value.
            if val.starts_with('-') && val != "-" {
                None
            } else {
                Some(val)
            }
        } else {
            None
        }
    })
}

/// Return the value of a flag using `=` syntax (e.g., `--output=/tmp/file`, `of=/tmp/out`).
fn flag_value_equals<'a>(parts: &'a [&'a str], prefix: &str) -> Option<&'a str> {
    parts.iter().find_map(|p| p.strip_prefix(prefix))
}

/// Check if `sed` has `-i` in a way that mutates files outside temp.
/// When all file operands after `-i` are under temp, returns `false` (allow).
fn has_sed_mutation(command: &str, state: &ValidationState) -> bool {
    let parts: Vec<&str> = command.split_whitespace().collect();
    // Find the -i flag position (including -i.bak, -iSUFFIX)
    let i_pos = parts
        .iter()
        .position(|p| *p == "-i" || p.starts_with("-i."));
    let Some(i_pos) = i_pos else {
        return false; // no -i flag → not a sed mutation
    };

    // Collect file operands after -i.
    // Skip flags (-n, -e, etc.), sed expressions, and backup-extension tokens.
    // Sed expressions and backup extensions are non-absolute tokens before the
    // first absolute-path file operand. Once we've seen a file operand, all
    // subsequent non-flag tokens are treated as file operands (multi-file).
    let mut file_operands: Vec<&str> = Vec::new();
    let mut seen_file_operand = false;

    for part in &parts[i_pos + 1..] {
        if part.starts_with('-') {
            continue; // skip flags like -n, -e, -e 'expr'
        }
        if seen_file_operand {
            // Already past the sed expression — this is a file operand
            file_operands.push(part);
            continue;
        }
        // Before the first file operand: skip non-absolute tokens (sed expression
        // like 's/a/b/' or backup extension like .bak for `-i .bak`).
        let p = Path::new(part);
        if p.is_absolute() {
            seen_file_operand = true;
            file_operands.push(part);
        }
        // Non-absolute tokens are sed expressions or backup extensions — skip
    }

    if file_operands.is_empty() {
        return true; // `sed -i` without file operand → reject (conservative)
    }

    !file_operands.iter().all(|p| {
        resolve_path_word(p, state).is_some_and(|path| is_path_under_temp(&path, state.ctx))
    })
}

/// Check if `awk -i inplace` is present.
fn has_inplace(command: &str, _state: &ValidationState) -> bool {
    let parts: Vec<&str> = command.split_whitespace().collect();
    parts.windows(2).any(|w| w[0] == "-i" && w[1] == "inplace")
}

/// Check if `dd of=...` writes outside temp.
/// When `of=` points to a temp path, returns `false` (allow).
fn has_dd_mutation(command: &str, state: &ValidationState) -> bool {
    let parts: Vec<&str> = command.split_whitespace().collect();
    if let Some(of_val) = flag_value_equals(&parts, "of=") {
        return !resolve_path_word(of_val, state)
            .is_some_and(|path| is_path_under_temp(&path, state.ctx));
    }
    false
}

/// Check if curl has output flags that write outside temp.
/// `-o <path>` / `--output <path>` is allowed when path is under temp.
/// `-O` / `--remote-name` is always blocked (writes to CWD with URL filename).
fn has_curl_mutation(command: &str, state: &ValidationState) -> bool {
    let parts: Vec<&str> = command.split_whitespace().collect();

    // -O/--remote-name are always blocked (writes to CWD with URL filename)
    if has_any_flag(command, &["-O", "--remote-name"]) {
        return true;
    }

    // -o/--output: blocked unless output path is under temp
    for flag in &["-o", "--output"] {
        if let Some(path) = flag_value(&parts, flag) {
            return !resolve_path_word(path, state)
                .is_some_and(|p| is_path_under_temp(&p, state.ctx));
        }
    }

    // --output=path syntax
    if let Some(path) = flag_value_equals(&parts, "--output=") {
        return !resolve_path_word(path, state).is_some_and(|p| is_path_under_temp(&p, state.ctx));
    }

    false
}

/// Check if `wget` output flags write outside temp.
/// `-O <path>` / `--output-document <path>` / `-P <path>` / `--directory-prefix <path>`
/// are allowed when the path is under temp.
/// Without output flags, wget writes to CWD → always blocked.
fn has_wget_mutation(command: &str, state: &ValidationState) -> bool {
    let parts: Vec<&str> = command.split_whitespace().collect();

    // -O/--output-document: specify output file path
    if let Some(path) = flag_value(&parts, "-O").or_else(|| flag_value(&parts, "--output-document"))
    {
        return !resolve_path_word(path, state).is_some_and(|p| is_path_under_temp(&p, state.ctx));
    }
    if let Some(path) = flag_value_equals(&parts, "--output-document=") {
        return !resolve_path_word(path, state).is_some_and(|p| is_path_under_temp(&p, state.ctx));
    }

    // -P/--directory-prefix: specify download directory
    if let Some(path) =
        flag_value(&parts, "-P").or_else(|| flag_value(&parts, "--directory-prefix"))
    {
        return !resolve_path_word(path, state).is_some_and(|p| is_path_under_temp(&p, state.ctx));
    }
    if let Some(path) = flag_value_equals(&parts, "--directory-prefix=") {
        return !resolve_path_word(path, state).is_some_and(|p| is_path_under_temp(&p, state.ctx));
    }

    // No known output flag → wget would write to CWD with URL's filename
    true
}

/// Characters that are non-operation tar flags (format/output modifiers).
/// These can appear alongside the operation flag in combined forms (e.g.
/// `-tvf` combines `t` (list) with `v` (verbose) and `f` (file)).
const TAR_SAFE_CHARS: &[char] = &['v', 'f', 'z', 'j', 'J'];

/// Check if tar is using only `-t`/`--list` (list) mode. Handles combined flags.
///
/// This is the **whitelist** for [`is_tar_mutating`]'s negative detection
/// strategy. Add new safe/list-only operations (e.g. `--diff`/`--compare`)
/// here rather than adding blacklist checks to [`is_tar_mutating`].
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

/// Check if `tar` is in mutating mode (i.e., will extract/create).
///
/// # Negative detection strategy
///
/// Unlike the other [`FLAG_CHECKS`] entries which use **positive detection**
/// (a predicate that returns `true` only when a specific mutating flag is
/// found), this function uses **negative detection**: it returns `true` when
/// the command is *not* explicitly list-only.
///
/// The reason is that `tar` combines short flags into strings (e.g. `-xvf`
/// or `-czf`), making it impractical to enumerate every mutating flag
/// combination. Instead, it's simpler to maintain a whitelist of safe
/// operations in [`is_tar_list_only`] and deny everything else.
///
/// The `!is_tar_list_only()` fallback automatically catches **all** mutating
/// operations — extraction (`-x`), creation (`-c`), appending (`-r`), update
/// (`-u`), deletion (`--delete`), etc. No individual blacklist checks are
/// needed.
///
/// ## For maintainers
///
/// *   **New mutating operations** require no changes — they are already
///     caught by the negative-detection fallback.
/// *   **New safe/list operations** (e.g. `--diff`/`--compare` for comparing
///     archive contents without modifying anything) should be added to
///     [`is_tar_list_only`] to whitelist them.
/// *   Do **not** add positive blacklist checks to this function — they
///     would be dead code, masked by the fallback.
fn is_tar_mutating(command: &str, _state: &ValidationState) -> bool {
    !is_tar_list_only(command)
}

/// Check if `base64 -d -o` writes outside temp.
/// When `-o`/`--output` points to a temp path, returns `false` (allow).
fn has_base64_mutation(command: &str, state: &ValidationState) -> bool {
    if !has_any_flag(command, &["-d", "--decode"]) {
        return false; // not decoding → not a mutation
    }

    let parts: Vec<&str> = command.split_whitespace().collect();

    // Check -o/--output flag values
    for flag in &["-o", "--output"] {
        if let Some(path) = flag_value(&parts, flag) {
            return !resolve_path_word(path, state)
                .is_some_and(|p| is_path_under_temp(&p, state.ctx));
        }
    }

    // No -o/--output flag → output to stdout, not mutating
    false
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

    /// Hermetic test context.
    ///
    /// The workspace root is a fixed absolute path that is NOT under any OS
    /// temp root (the "hermetic-test trap": a fixture workspace created under
    /// `tempfile::tempdir()` lies inside the allowed temp roots, making
    /// 'workspace write blocked' assertions impossible). Relative paths
    /// resolve against this root and are therefore rejected as workspace
    /// writes, matching production semantics where the workspace lives outside
    /// the OS temp dirs.
    fn test_ctx() -> CheckContext {
        CheckContext {
            workspace_root: std::path::PathBuf::from("/__mahbot_readonly_test_ws__"),
            temp_roots: crate::tools::path::allowed_temp_roots(),
            // Match the session shell environment (TMPDIR=/tmp); TMP/TEMP unset.
            temp_vars: vec![("TMPDIR".to_string(), "/tmp".to_string())],
        }
    }

    fn ok(cmd: &str) {
        let ctx = test_ctx();
        assert!(
            check_command(cmd, &ctx).is_ok(),
            "expected ALLOW but got REJECT for: `{cmd}`"
        );
    }

    fn assert_rejected(cmd: &str) {
        let ctx = test_ctx();
        assert!(
            check_command(cmd, &ctx).is_err(),
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

    /// Assert each token-classification case in a table-driven test.
    fn run_token_cases(cases: &[(&str, TokenKind)]) {
        for (input, expected) in cases {
            assert_eq!(classify_shell_token(input), *expected, "{input:?}");
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
            // branch/tag name-creation bypass (regression tests)
            ("git branch my-feature", false),
            ("git branch --force my-feature", false),
            ("git branch -f my-feature", false),
            ("git tag v1.0", false),
            ("git tag -f v1.0", false),
            ("git tag --force v1.0", false),
            ("git branch -u origin/main", false),
            ("git branch --set-upstream-to=origin/main", false),
            // --track bypass (flag as first arg, not in name-creation check)
            ("git branch --track feature", false),
            ("git branch --no-track feature", false),
            // tag message flag bypasses
            ("git tag -m msg v1.0", false),
            ("git tag --message msg v1.0", false),
            ("git tag -F file v1.0", false),
            ("git tag --file file v1.0", false),
            ("git tag -e v1.0", false),
            ("git tag --edit v1.0", false),
            // remote inspection commands
            ("git remote", true),
            ("git remote -v", true),
            ("git remote show origin", true),
            ("git remote get-url origin", true),
            // mktag always mutates
            ("git mktag <tag_object", false),
            ("git mktag", false),
            // mktree always mutates
            ("git mktree <tree_contents", false),
            ("git mktree", false),
            // merge-file always mutates (removed from safe list)
            ("git merge-file a.txt b.txt c.txt", false),
            ("git merge-file -p a.txt b.txt c.txt", false),
            // merge-tree always mutates (removed from safe list)
            ("git merge-tree base branch1 branch2", false),
            ("git merge-tree --write-tree base branch1 branch2", false),
            // hash-object: read-only without -w, mutating with -w
            ("git hash-object file.txt", true),
            ("git hash-object --stdin", true),
            ("git hash-object -w file.txt", false),
            ("git hash-object -w --stdin", false),
            ("git hash-object file.txt -w", false),
            // reflog: read-only commands allowed, expire/delete rejected
            ("git reflog", true),
            ("git reflog show", true),
            ("git reflog list", true),
            ("git reflog show HEAD", true),
            ("git reflog expire --all", false),
            ("git reflog delete HEAD@{0}", false),
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
            // cargo update and generate-lockfile are rejected with tailored messages
            ("cargo update", false),
            ("cargo update --dry-run", true), // dry-run preview — read-only
            ("cargo generate-lockfile", false),
        ];

        run_cases(&cases);
    }

    // ── Unconditional rejections ──────────────────────────────────

    /// Tests that ALL entries in the production [`MUTATING_COMMANDS`] constant
    /// are rejected. Iterates the constant directly to prevent coverage drift
    /// when entries are added or removed.
    #[test]
    fn all_mutating_commands_rejected() {
        // Use an absolute non-temp path to prevent accidental test drift
        // if items are later moved to TEMP_MUTATORS.
        assert_all_rejected(MUTATING_COMMANDS, |cmd| format!("{cmd} /etc/blocked_test"));
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

    /// Tests that safe branch/tag listing commands are accepted (non-regression
    /// for the name-creation bypass fix — these must NOT be blocked).
    #[test]
    fn git_branch_tag_safe_listing() {
        let cases = [
            // branch: no args (list local branches)
            ("git branch", true),
            // branch: standard listing flags
            ("git branch --list", true),
            ("git branch --list feature", true),
            ("git branch -l", true),
            ("git branch -l feature", true),
            ("git branch --merged", true),
            ("git branch --merged main", true),
            ("git branch --no-merged", true),
            ("git branch --no-merged main", true),
            ("git branch -a", true),
            ("git branch -r", true),
            ("git branch -v", true),
            ("git branch -vv", true),
            ("git branch --sort=-committerdate", true),
            ("git branch --contains abc123", true),
            ("git branch --points-at abc123", true),
            ("git branch --format='%(refname)'", true),
            // tag: no args (list tags)
            ("git tag", true),
            // tag: standard listing flags
            ("git tag --list", true),
            ("git tag -l", true),
            ("git tag -l v1*", true),
            ("git tag --contains abc123", true),
            ("git tag --merged main", true),
            ("git tag --points-at abc123", true),
            ("git tag -n", true),
        ];

        run_cases(&cases);
    }

    /// Tests that mutation flags in later argument positions are correctly caught,
    /// and that bare-word remote verbs in position 2+ after a safe sub-subcommand
    /// are correctly allowed (no false positives).
    #[test]
    fn git_mutation_in_later_position() {
        let cases = [
            // branch: mutation flag -d in position 2 (after a safe flag)
            ("git branch --sort=-committerdate -d", false),
            // branch: mutation flag --delete in position 2 (after a safe flag)
            ("git branch --sort=-committerdate --delete feature", false),
            // tag: mutation flag --delete in position 2 (after -l and pattern)
            ("git tag -l v1.* --delete", false),
            // tag: mutation flag -d in position 2
            ("git tag -l v1.* -d", false),
            // remote: bare-word "add" in position 2 after safe sub-subcommand
            // should be ALLOWED (it's a remote name, not a mutation verb)
            ("git remote show add", true),
            // remote: bare-word verb "update" after a flag should be REJECTED
            // (first non-flag argument)
            ("git remote -v update", false),
        ];

        run_cases(&cases);
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

    // ── classify_shell_token unit tests ──────────────────────────────

    /// Verify [`classify_shell_token`] returns the correct [`TokenKind`] for all
    /// operator variants, plain tokens, and ordering-sensitive edge cases.
    ///
    /// See the function's own doc comment for the ordering-invariant rationale.
    ///
    /// **Note:** All cases share a single `#[test]` entry point, so a panic
    /// in any case aborts the remaining cases in that run. This is an accepted
    /// trade-off for keeping fast, pure-function test data in one place.
    #[test]
    fn classify_shell_token_table() {
        const NO_TARGET: TokenKind = TokenKind::Redirect {
            needs_target: false,
        };

        run_token_cases(&[
            // ── Standalone output redirect (expects target) ────────
            (">", TokenKind::Redirect { needs_target: true }),
            (">&", TokenKind::Redirect { needs_target: true }),
            (">>", TokenKind::Redirect { needs_target: true }),
            (">|", TokenKind::Redirect { needs_target: true }),
            // ── Standalone input redirect (expects target) ─────────
            ("<", TokenKind::Redirect { needs_target: true }),
            ("<&", TokenKind::Redirect { needs_target: true }),
            ("<>", TokenKind::Redirect { needs_target: true }),
            // ── Digit-prefixed standalone (expects target) ─────────
            ("2>", TokenKind::Redirect { needs_target: true }),
            ("10>", TokenKind::Redirect { needs_target: true }),
            ("3<", TokenKind::Redirect { needs_target: true }),
            // ── Self-contained fd-merge (no target) ────────────────
            ("2>&1", NO_TARGET),
            ("1>&2", NO_TARGET),
            // ── Bash standalone (expects target) ───────────────────
            ("&>", TokenKind::Redirect { needs_target: true }),
            ("&>>", TokenKind::Redirect { needs_target: true }),
            // ── Heredoc variants (self-contained; segments are always
            //    heredoc-stripped, so this is defensive) ────────────
            ("<<EOF", NO_TARGET),
            ("<<-EOF", NO_TARGET),
            ("<<'EOF'", NO_TARGET),
            ("3<<EOF", NO_TARGET),
            ("1<<-EOF", NO_TARGET),
            // ── Herestrings (no body — redirect-like, consumes next word) ─
            ("<<<", TokenKind::Redirect { needs_target: true }),
            ("<<<hi", TokenKind::Redirect { needs_target: true }),
            ("3<<<", TokenKind::Redirect { needs_target: true }),
            // ── Combined redirect (operator + target, no skip) ─────
            (">/dev/null", NO_TARGET),
            (">>file", NO_TARGET),
            ("</dev/null", NO_TARGET),
            ("<&2", NO_TARGET),
            ("<>file", NO_TARGET),
            ("2>/dev/null", NO_TARGET),
            ("1>/tmp/out", NO_TARGET),
            ("3</dev/null", NO_TARGET),
            ("&>/dev/null", NO_TARGET),
            ("&>>file", NO_TARGET),
            (">&2", NO_TARGET),
            // ── Non-redirect tokens (plain) ────────────────────────
            ("hello", TokenKind::Regular),
            ("file.txt", TokenKind::Regular),
            ("path/to/file", TokenKind::Regular),
            ("test", TokenKind::Regular),
            // ── Flags (not redirects) ──────────────────────────────
            ("-o", TokenKind::Regular),
            ("--output", TokenKind::Regular),
            ("-f", TokenKind::Regular),
            ("--force", TokenKind::Regular),
            ("-d", TokenKind::Regular),
            // ── Bare digits (not redirects) ────────────────────────
            ("2", TokenKind::Regular),
            ("10", TokenKind::Regular),
            ("3", TokenKind::Regular),
        ]);
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

    /// Tests that `<<` inside quotes is not treated as a heredoc start.
    /// Without quote-state tracking in
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
                name: "stderr capture suffix skipped",
                input: "git --version 2>&1",
                expected: "",
            },
            Case {
                name: "stderr capture after subcommand",
                input: "git status 2>&1",
                expected: "status 2>&1",
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
            ("rm /tmp/scratch.txt", true),
            // ── TEMP_MUTATORS (cp, mv, rm, gzip, etc.) ──
            ("cp /tmp/a /tmp/b", true),
            ("mv /tmp/a /tmp/b", true),
            ("cp /tmp/a /etc/passwd", false),
            ("mv /tmp/a /etc/passwd", false),
            ("rm /etc/passwd", false),
            ("rmdir /tmp/scratch_dir", true),
            ("gzip /tmp/file.txt", true),
            ("gunzip /tmp/file.txt.gz", true),
            ("bzip2 /tmp/file.txt", true),
            ("xz /tmp/file.txt", true),
            ("zstd /tmp/file.txt", true),
            ("zip /tmp/out.zip /tmp/file1 /tmp/file2", true),
            // cp: only the DESTINATION must be under temp (sources are
            // read-only — copying from anywhere into temp is allowed;
            // copying into the workspace stays blocked, decision 8).
            ("cp /etc/passwd /tmp/out", true), // source outside temp, dest temp
            // ── Flag-based temp-aware checks ──
            // curl -o to temp → allowed
            ("curl -o /tmp/file URL", true),
            ("curl --output /tmp/file URL", true),
            ("curl -o /etc/passwd URL", false),
            ("curl --output /etc/passwd URL", false),
            // curl -O/--remote-name always blocked (writes to CWD)
            ("curl -O URL", false),
            ("curl --remote-name URL", false),
            // curl -o with a mix: -o OK but -O still blocked
            ("curl -o /tmp/file -O URL", false), // -O always blocks
            // wget -O to temp → allowed
            ("wget -O /tmp/file URL", true),
            ("wget --output-document /tmp/file URL", true),
            ("wget -O /etc/passwd URL", false),
            ("wget --output-document /etc/passwd URL", false),
            ("wget -P /tmp/dir URL", true),
            ("wget --directory-prefix /tmp/dir URL", true),
            ("wget -P /etc/dir URL", false),
            // wget without output flags → always blocked
            ("wget URL", false),
            // sed -i to temp → allowed
            ("sed -i 's/a/b/' /tmp/file", true),
            ("sed -i.bak 's/a/b/' /tmp/file", true),
            ("sed -i 's/a/b/' /tmp/file1 /tmp/file2", true),
            ("sed -i 's/a/b/' /etc/passwd", false),
            ("sed -i 's/a/b/' /tmp/file /etc/passwd", false), // mixed
            ("sed -i '' /tmp/file", true),                    // empty backup ext
            // sed -i with -e flag
            ("sed -i -e 's/a/b/' /tmp/file", true),
            ("sed -i -e 's/a/b/' /etc/passwd", false),
            // sed -i with separate backup extension
            ("sed -i .bak 's/a/b/' /tmp/file", true),
            ("sed -i .bak 's/a/b/' /etc/passwd", false),
            // dd of= to temp → allowed
            ("dd if=/dev/zero of=/tmp/out bs=1024 count=1", true),
            ("dd of=/tmp/out", true),
            ("dd if=/dev/zero of=/etc/passwd bs=1024 count=1", false),
            ("dd of=/etc/passwd", false),
            // base64 -d -o to temp → allowed
            ("base64 -d -o /tmp/out input.txt", true),
            ("base64 -d --output /tmp/out input.txt", true),
            ("base64 -d -o /etc/passwd input.txt", false),
            ("base64 -d --output /etc/passwd input.txt", false),
            // base64 -d without -o → stdout, always allowed
            ("base64 -d input.txt", true),
            // base64 with -o but without -d → no mutation
            ("base64 -o /tmp/out input.txt", true),
            // ── Multiple path arguments (security bypass) ──
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

    /// Tests that ALL entries in [`TEMP_MUTATORS`] are allowed with temp paths
    /// and rejected with non-temp paths, preventing coverage drift.
    #[test]
    fn all_temp_mutators_allowed_with_temp_paths() {
        // cp, mv, rm, rmdir, unlink: basic file ops
        let temp_ops: &[(&str, &str)] = &[
            ("cp", "/tmp/a /tmp/b"),
            ("mv", "/tmp/a /tmp/b"),
            ("rm", "/tmp/scratch.txt"),
            ("rmdir", "/tmp/scratch_dir"),
            ("unlink", "/tmp/link"),
            ("gzip", "/tmp/file.txt"),
            ("gunzip", "/tmp/file.txt.gz"),
            ("bzip2", "/tmp/file.txt"),
            ("xz", "/tmp/file.txt"),
            ("zstd", "/tmp/file.txt"),
            ("zip", "/tmp/out.zip /tmp/file1"),
        ];
        for &(cmd, args) in temp_ops {
            ok(&format!("{cmd} {args}"));
        }
    }

    /// Tests that ALL entries in [`TEMP_MUTATORS`] are rejected when used
    /// with a non-temp absolute path, preventing accidental test drift.
    #[test]
    fn all_temp_mutators_rejected_with_non_temp() {
        let temp_ops: &[&str] = &[
            "cp", "mv", "rm", "rmdir", "unlink", "gzip", "gunzip", "bzip2", "xz", "zstd", "zip",
        ];
        for &cmd in temp_ops {
            assert_rejected(&format!("{cmd} /etc/blocked_test"));
        }
    }

    // ── Phase 1 acceptance: heredocs ──────────────────────────────────────

    /// Every fixed heredoc pattern AND its mutating sibling (guard contract,
    /// phase 1.1).
    #[test]
    fn heredoc_acceptance() {
        let cases = [
            // Write command immediately after the terminator must NOT escape
            // scanning (live bypass: leaked_guard_test.txt).
            ("cat <<EOF\nbody\nEOF\ntouch workspace_file", false),
            ("cat <<EOF\nbody\nEOF\ntouch /tmp/scratch_file", true),
            // Write on the delimiter line must be scanned (live bypass).
            ("cat <<EOF > workspace_file", false),
            ("cat <<EOF > /tmp/out", true),
            // Heredoc body containing mutator-shaped text is literal — allowed.
            ("cat <<EOF\nrm -rf /tmp\nEOF", true),
            ("cat <<EOF\ncat > workspace_file\nEOF", true),
            // Two-heredoc chain: bodies excluded, delimiter-line redirect scanned.
            (
                "cat <<EOF1 <<EOF2 > /tmp/out\nbody1 > x\nEOF1\nbody2 > y\nEOF2",
                true,
            ),
            // <<- with tab-indented terminator.
            ("cat <<-EOF\n\tbody\n\tEOF", true),
            // <<- strips ALL leading tabs (bash semantics): a 2+-tab-indented
            // terminator still terminates, so a write on the line after it
            // must be scanned (QA round: the matcher stripped only one tab,
            // swallowing the write as body text).
            ("cat <<-EOF\n\tbody\n\t\tEOF", true),
            ("cat <<-EOF\n\tbody\n\t\tEOF\ntouch workspace_file", false),
            ("cat <<-EOF\n\tbody\n\t\t\tEOF\ntouch workspace_file", false),
            ("cat <<-EOF\n\tbody\n\t\tEOF\ntouch /tmp/scratch_file", true),
            // CRLF line endings: write after terminator still blocked.
            ("cat <<EOF\r\nbody\r\nEOF\r\ntouch workspace_file", false),
            ("cat <<EOF\r\nbody\r\nEOF", true),
            // Terminator at end of input (no trailing newline).
            ("cat <<EOF\nbody\nEOF", true),
            // Quoted delimiter.
            ("cat <<'EOF'\nbody\nEOF", true),
            // Heredoc with workspace write after, chained with && on the
            // line after the terminator.
            ("cat <<EOF\nbody\nEOF\n&& touch workspace_file", false),
            // Double-quoted `$()` in an unquoted heredoc body still expands
            // in bash (quote chars are literal in heredoc bodies) — blocked.
            ("cat <<EOF\n\"$(touch workspace_file)\"\nEOF", false),
            // Multi-tab <<- terminator with a substitution body and a write
            // after the terminator — all caught.
            (
                "cat <<-EOF\n\t$(touch workspace_file)\n\t\tEOF\ntouch /tmp/ok",
                false,
            ),
            // A command on the terminator line itself is NOT a terminator in
            // bash (unterminated heredoc — nothing executes), so it must not
            // be treated as a command.
            ("cat <<EOF\nbody\nEOF && touch workspace_file", true),
            // fd-prefixed heredoc body not scanned as command text.
            ("tee /tmp/x 3<< 'EOF'\nbody\nEOF", true),
            // ── Command substitutions in heredoc bodies ─────────────
            // Real bash executes `$(...)`/backticks inside UNQUOTED heredoc
            // bodies, so those spans must remain scanned (reviewer round).
            ("cat <<EOF\n$(touch workspace_file)\nEOF", false),
            ("cat <<EOF\n$(echo hi > workspace_file)\nEOF", false),
            ("cat <<EOF\n`touch workspace_file`\nEOF", false),
            // Temp-targeted body substitutions stay allowed.
            ("cat <<EOF\n$(touch /tmp/x)\nEOF", true),
            ("cat <<EOF\n$(echo hi > /tmp/out)\nEOF", true),
            // Nested substitution in an unquoted body.
            ("cat <<EOF\n$(echo $(touch workspace_file))\nEOF", false),
            // Escaped `\$(` in an unquoted body is literal — allowed.
            ("cat <<EOF\n\\$(touch workspace_file)\nEOF", true),
            // Quoted delimiters make the body literal — no expansion.
            ("cat <<'EOF'\n$(touch workspace_file)\nEOF", true),
            ("cat <<\"EOF\"\n$(touch workspace_file)\nEOF", true),
            // Plain mutator-shaped body text (no substitution) stays literal.
            ("cat <<EOF\ntouch workspace_file\nEOF", true),
            // Multiple heredocs: a body substitution in the first body is
            // still scanned.
            ("cat <<A <<B\n$(touch workspace_file)\nA\nB", false),
            // Body substitution followed by a workspace write after the
            // terminator — both caught.
            ("cat <<EOF\n$(echo hi)\nEOF\ntouch workspace_file", false),
            // Body substitution under a temp cd — allowed (state at the
            // heredoc's position in the chain applies).
            ("cd /tmp && cat <<EOF\n$(touch rel)\nEOF", true),
        ];

        run_cases(&cases);
    }

    // ── Phase 1 acceptance: herestrings ──────────────────────────────────

    /// `<<<` must not hide a following redirect (live bypass:
    /// herestring_escape_test.txt).
    #[test]
    fn herestring_acceptance() {
        let cases = [
            ("cat <<< hi", true),
            ("cat <<< hi > workspace_file", false),
            ("cat <<< hi > /tmp/out", true),
            ("cat <<< hi > /dev/null", true),
            ("cat <<< \"hi there\" > /tmp/out", true),
            ("tee /tmp/scratch.log <<< hi", true),
        ];

        run_cases(&cases);
    }

    // ── Phase 1 acceptance: substitutions ────────────────────────────────

    /// `$(...)` and backtick contents are validated as nested commands.
    #[test]
    fn substitution_acceptance() {
        let cases = [
            // Suppressed-stderr read inside a substitution — allowed.
            ("echo $(cat /etc/passwd 2>/dev/null)", true),
            ("echo `cat /etc/passwd 2>/dev/null`", true),
            // Write inside a substitution — blocked.
            ("echo $(touch workspace_file)", false),
            ("echo `touch workspace_file`", false),
            // Redirect to temp inside a substitution — allowed.
            ("echo $(echo hi > /tmp/out)", true),
            // Redirect to workspace inside a substitution — blocked.
            ("echo $(echo hi > out.txt)", false),
            ("echo $(echo hi > /etc/passwd)", false),
            // Nested substitution.
            ("echo $(echo $(rm -rf /tmp/x))", true),
            ("echo $(echo $(touch ws_file))", false),
            // Substitution with a mutating command under temp — allowed.
            ("echo $(rm -rf /tmp/scratch_dir)", true),
            // Substitution that writes to workspace via a temp-qualified var is
            // rejected because the relative target resolves to the workspace.
            ("echo $(tee /tmp/ok > ws_out)", false),
        ];

        run_cases(&cases);
    }

    // ── Phase 1.3 acceptance: double-quoted substitutions ────────────────

    /// bash executes `$(...)`/backticks inside double quotes
    /// (`echo "$(touch f)"` runs `touch f`), so writes hidden in
    /// double-quoted substitutions must be rejected (QA round: the guard
    /// previously skipped all double-quoted content, leaving this spelling
    /// of the phase-1.3 bypass open). Literal quoted text (no substitution)
    /// and escaped/single-quoted `$(` stay allowed.
    #[test]
    fn double_quoted_substitution_acceptance() {
        let cases = [
            // Flat double-quoted substitution with a workspace write — blocked.
            ("echo \"$(touch workspace_file)\"", false),
            ("echo \"$(echo hi > workspace_file)\"", false),
            ("echo \"`touch workspace_file`\"", false),
            ("echo \"$(rm -f workspace_file)\"", false),
            ("echo \"$(rm -f /tmp/scratch_file)\"", true),
            // Nested double-quoted substitution — blocked at the inner level.
            ("echo \"$(echo $(touch nested))\"", false),
            ("echo \"$(echo hi > $(echo workspace_file))\"", false),
            // Temp-targeted double-quoted substitution — allowed.
            ("echo \"$(echo hi > /tmp/out)\"", true),
            ("echo \"$(touch /tmp/scratch)\"", true),
            // Plain double-quoted substitution (pure read) — allowed.
            ("echo \"$(echo hi)\"", true),
            ("echo \"$(ls 2>&1)\"", true),
            // Literal double-quoted text with a `>` — not a redirect.
            ("echo \"a > b\"", true),
            // Outside-quote redirect after a double-quoted substitution.
            ("echo \"$(echo hi)\" > /tmp/out", true),
            ("echo \"$(echo hi)\" > workspace_file", false),
            // Escaped `\$(` inside double quotes is literal — allowed.
            ("echo \"abc\\$(touch workspace_file)\"", true),
            // Escaped backtick inside double quotes is literal — allowed.
            ("echo \"abc\\`touch workspace_file\\`\"", true),
            // Single-quoted `$(` is literal — allowed.
            ("echo '$(touch workspace_file)'", true),
            // Backtick substitution spanning a quote-aware body.
            ("echo \"`echo hi; echo hi2`\"", true),
            // Double-quoted substitution with an internal cd — state applies.
            ("echo \"$(cd /tmp; touch rel)\"", true),
            ("echo \"$(cd /etc; touch rel)\"", false),
            // Poisoned export in a prior segment applies to a double-quoted
            // substitution (decision 5 + phase 1.3 integration).
            ("export TMPDIR=/etc\necho \"$(touch $TMPDIR/x)\"", false),
            (
                "export TMPDIR=/__mahbot_readonly_test_ws__\necho \"$(echo hi > $TMPDIR/out)\"",
                false,
            ),
            (
                "export TMPDIR=/etc\nexport TMPDIR=/tmp\necho \"$(touch $TMPDIR/x)\"",
                true,
            ),
            // Herestring content expands substitutions — a write in a
            // double-quoted herestring word must be caught.
            ("cat <<< \"$(touch workspace_file)\"", false),
            ("cat <<< \"$(echo hi)\"", true),
        ];

        run_cases(&cases);
    }

    // ── Phase 1.3×2 integration: substitution × cd/export state ──────────

    /// Substitutions must be validated with the state at their segment's
    /// position — prior segments' `cd`/export bindings apply (reviewer round:
    /// the upfront whole-string scan validated every substitution against the
    /// initial state, leaving the poisoned-export bypass open and rejecting
    /// legit temp writes after a `cd`).
    #[test]
    fn substitution_state_acceptance() {
        let cases = [
            // Export-poisoning bypass: `export TMPDIR=/etc` in a prior segment
            // must poison expansions inside a later substitution (phase 1.3
            // "verifiably closed" + decision 5).
            ("export TMPDIR=/etc\necho $(touch $TMPDIR/x)", false),
            ("export TMPDIR=/etc\ncat $(echo hi > $TMPDIR/out)", false),
            // Same via the workspace-root spelling.
            (
                "export TMPDIR=/__mahbot_readonly_test_ws__\necho $(echo hi > $TMPDIR/out)",
                false,
            ),
            // Env-prefix binding in a prior segment poisons a later
            // substitution (decision 5's env-prefix form).
            ("TMPDIR=/etc true\necho $(touch $TMPDIR/x)", false),
            // Poison cleared by a temp-root rebind before the substitution.
            (
                "export TMPDIR=/etc\nexport TMPDIR=/tmp\necho $(touch $TMPDIR/x)",
                true,
            ),
            // cd to temp + relative write inside a substitution — allowed
            // (false-positive fix: the upfront scan rejected this against the
            // workspace-root CWD).
            ("cd /tmp && echo $(touch rel)", true),
            ("cd /tmp && echo $(echo hi > out.txt)", true),
            // cd to a non-temp dir + relative write inside a substitution —
            // blocked.
            ("cd /etc && echo $(touch rel)", false),
            // Substitution-internal `;` must not fragment the substitution:
            // a `cd` inside the subshell applies to the rest of the
            // substitution content.
            ("echo $(cd /tmp; touch rel)", true),
            ("echo $(cd /etc; touch rel)", false),
            // Export inside a substitution poisons only the substitution's
            // own environment (snapshot semantics), not the outer command.
            ("echo $(export TMPDIR=/etc; touch $TMPDIR/x)", false),
            (
                "echo $(export TMPDIR=/etc; echo hi)\ntouch $TMPDIR/outer",
                true,
            ),
            // Mutator inside a substitution after a temp cd stays blocked
            // when it targets the workspace (relative to the subshell CWD).
            ("cd /tmp && echo $(touch ws_file)", true),
            // A workspace write AFTER a substitution is still caught.
            ("cd /tmp && echo $(touch rel) && touch ws_file", true),
            ("echo $(echo hi) && touch ws_file", false),
        ];

        run_cases(&cases);
    }

    // ── Phase 1 acceptance: redirect-target delimiters ───────────────────

    #[test]
    fn redirect_delimiter_acceptance() {
        let cases = [
            // `)` terminates a redirect target (suppressed-stderr read).
            ("echo $(cmd 2>/dev/null)", true),
            ("echo $(cmd 2>/dev/null) done", true),
            // Redirect to /etc inside a substitution — blocked.
            ("echo $(echo x > /etc/passwd)", false),
            // Temp-file redirect ending in `)` — allowed.
            ("echo $(echo x > /tmp/out)", true),
            // Relative-file redirect ending in `)` — blocked.
            ("echo $(echo x > out.txt)", false),
            // `}` terminates a redirect target too (brace-closed groups).
            ("echo $(cmd 2>/dev/null} done", true),
        ];

        run_cases(&cases);
    }

    // ── Phase 1 acceptance: newline as command separator ─────────────────

    #[test]
    fn newline_separator_acceptance() {
        let cases = [
            // Multi-line temp setup script: each line parses as its own command.
            ("touch /tmp/a\necho hi > /tmp/b\ncat /tmp/a", true),
            // Workspace write across a newline — blocked.
            ("echo hi\ntouch workspace_file", false),
            ("touch workspace_file\necho hi", false),
            // Heredoc body with mutator-shaped text is not a command.
            ("cat <<EOF\nrm -rf /tmp\nEOF\necho done", true),
            // Backslash-newline continuation joins the logical line.
            ("echo hello \\\nworld", true),
            ("touch /tmp/a \\\n/tmp/b", true),
            // Quote-aware newline: quoted newline is not a separator.
            ("echo 'a\nb'", true),
        ];

        run_cases(&cases);
    }

    // ── Phase 2 acceptance: variable/tilde/quote expansion ───────────────

    #[test]
    fn expansion_acceptance() {
        let cases = [
            // Temp writes via env vars and quoted env vars — allowed.
            ("touch $TMPDIR/out.txt", true),
            ("touch \"$TMPDIR/out.txt\"", true),
            ("touch ${TMPDIR}/out.txt", true),
            ("echo hi > $TMPDIR/out", true),
            ("echo hi > \"$TMPDIR/out\"", true),
            ("tee $TMPDIR/scratch.log", true),
            ("mkdir -p $TMPDIR/scratch_dir", true),
            ("rm $TMPDIR/scratch.txt", true),
            ("sed -i 's/a/b/' /tmp/file", true),
            ("dd of=$TMPDIR/out bs=1 count=1", true),
            ("curl -o $TMPDIR/file URL", true),
            // $PWD resolves to the workspace (initial CWD) — blocked.
            ("touch $PWD/out.txt", false),
            ("echo hi > $PWD/out", false),
            // Unknown/unset variables — blocked.
            ("touch $FOO/out.txt", false),
            ("touch $TMP/out.txt", false), // TMP unset in the session env
            // Tilde / $HOME — never under temp — blocked.
            ("touch ~/out.txt", false),
            ("touch $HOME/out.txt", false),
            // Single-quoted (unexpanded) var — blocked.
            ("touch '$TMPDIR/out.txt'", false),
            // Unbalanced quotes — blocked.
            ("touch \"$TMPDIR/out.txt", false),
        ];

        run_cases(&cases);
    }

    // ── Phase 2 acceptance: export poisoning ─────────────────────────────

    #[test]
    fn export_poisoning_acceptance() {
        let cases = [
            // Binding TMPDIR to a non-temp path poisons it (export form).
            ("export TMPDIR=/etc\necho hi > $TMPDIR/out", false),
            // Env-prefix form poisons too.
            ("TMPDIR=/etc touch $TMPDIR/out", false),
            // Plain assignment segment poisons.
            ("TMPDIR=/etc\ntouch $TMPDIR/out", false),
            // Re-binding to a valid temp root clears the poison.
            (
                "export TMPDIR=/etc\nexport TMPDIR=/tmp\ntouch $TMPDIR/out",
                true,
            ),
            // Env-prefix rebind to temp clears.
            ("TMPDIR=/etc\nTMPDIR=/tmp touch $TMPDIR/out", true),
            // Binding TMP (initially unset) to temp enables it.
            ("export TMP=/tmp\ntouch $TMP/out", true),
            // Bare export / export NAME are no-ops.
            ("export\ntouch $TMPDIR/out", true),
            ("export TMP\ntouch $TMPDIR/out", true),
            // Quoted value in export.
            ("export TMPDIR=\"/tmp\"\ntouch $TMPDIR/out", true),
        ];

        run_cases(&cases);
    }

    // ── Phase 2 acceptance: copy vs move/delete semantics ────────────────

    #[test]
    fn copy_move_acceptance() {
        let cases = [
            // cp: only the destination must be under temp.
            ("cp /etc/passwd /tmp/out", true),
            ("cp /tmp/a /tmp/b", true),
            ("cp -t /tmp/d s1 s2", true),
            ("cp --target-directory /tmp/d s1 s2", true),
            ("cp --target-directory=/tmp/d s1", true),
            // cp into the workspace stays blocked.
            ("cp /etc/passwd ws_file", false),
            ("cp /tmp/a /etc/passwd", false),
            ("cp -t /workspace/d s1", false),
            ("cp", false),
            // mv/rm: all args must be under temp (move from workspace to temp
            // deletes a source → blocked).
            ("mv /tmp/a /tmp/b", true),
            ("mv ws_file /tmp/out", false),
            ("rm /tmp/scratch.txt", true),
            ("rm ws_file", false),
        ];

        run_cases(&cases);
    }

    // ── Phase 2 acceptance: current-directory tracking ───────────────────

    #[test]
    fn cwd_tracking_acceptance() {
        let cases = [
            // cd to temp + relative write — allowed.
            ("cd /tmp && touch f", true),
            ("cd /tmp && echo hi > out.txt", true),
            ("cd /tmp && tee scratch.log", true),
            // cd to a non-temp dir then relative write — blocked.
            ("cd /tmp && cd /etc && touch f", false),
            ("cd / && touch f", false),
            // Relative cd under a tracked temp cwd tracks iff the normalized
            // result stays under a temp root AND the directory exists or was
            // created by `mkdir` earlier in the chain (a nonexistent target
            // fails the `cd` at runtime, so tracking its deeper path would let
            // a `..` chained write escape the temp root).
            ("mkdir -p /tmp/src && cd /tmp && cd src && touch f", true),
            (
                "mkdir -p /tmp/a/b/c && cd /tmp && cd a/b/c && echo hi > out.txt",
                true,
            ),
            ("cd /tmp && cd .. && touch f", false),
            ("cd /tmp && cd ../etc && touch f", false),
            ("cd /tmp && cd ../../../etc && touch f", false),
            ("cd /tmp && cd /etc && cd src && touch f", false),
            // Nonexistent relative targets fail the `cd` at runtime — the
            // chained write lands one level shallower, where a `..` chain
            // escapes the temp root. Fail closed in every chaining spelling.
            ("cd /tmp && cd a/b/c && touch ../x", false),
            ("cd /tmp ; cd a/b/c ; touch ../x", false),
            ("cd /tmp\ncd a/b/c\ntouch ../x", false),
            (
                "mkdir -p /tmp/a/b/c && cd /tmp && cd a/b/c && touch ../x",
                true,
            ),
            ("cd .. && touch f", false),
            ("cd - && touch f", false),
            ("cd && touch f", false),
            ("cd $TMPDIR && touch f", false),
            ("cd ~ && touch f", false),
            // pushd/popd always reset.
            ("cd /tmp && pushd /tmp && touch f", false),
            ("cd /tmp && popd && touch f", false),
            // cd to nonexistent dir + write — blocked (non-temp target).
            ("cd /nonexistent_dir_xyz_1059 && touch f", false),
            // cd to a nonexistent temp dir fails closed in every chaining
            // spelling: at runtime the cd fails and the write lands in the
            // real CWD (the workspace). The dir must exist at validation time
            // or have been created by `mkdir` earlier in the same chain.
            ("cd /tmp/nonexistent_x && touch f", false),
            ("cd /tmp/nonexistent_x ; touch f", false),
            ("cd /tmp/nonexistent_x\ntouch f", false),
            ("cd /tmp/nonexistent_x || touch f", false),
            ("mkdir -p /tmp/snap && cd /tmp/snap && touch f", true),
            ("mkdir -p /tmp/snap; cd /tmp/snap; touch f", true),
            ("mkdir -p /tmp/snap\ncd /tmp/snap\ntouch f", true),
            ("mkdir /tmp/snap; cd /tmp/snap; touch f", true),
            // mkdir -p records ancestors too.
            ("mkdir -p /tmp/a/b; cd /tmp/a; touch f", true),
            // Non-`-p` mkdir into a missing parent errors at runtime — the
            // target is not provably created, so the chained cd/write fails
            // closed in every spelling.
            (
                "mkdir /tmp/absent_parent_x/dir_y; cd /tmp/absent_parent_x/dir_y; touch f",
                false,
            ),
            (
                "mkdir /tmp/absent_parent_x/dir_y\ncd /tmp/absent_parent_x/dir_y\ntouch f",
                false,
            ),
            (
                "mkdir /tmp/absent_parent_x/dir_y || cd /tmp/absent_parent_x/dir_y || touch f",
                false,
            ),
            (
                "mkdir /tmp/absent_parent_x/dir_y; cd /tmp/absent_parent_x/dir_y; rm -rf x",
                false,
            ),
            (
                "mkdir /tmp/absent_parent_x/dir_y; cd /tmp/absent_parent_x/dir_y; tee out",
                false,
            ),
            // A non-`-p` target whose parent was created earlier in the chain
            // (transitively) is provable.
            (
                "mkdir -p /tmp/chain_a; mkdir /tmp/chain_a/chain_b; cd /tmp/chain_a/chain_b; touch f",
                true,
            ),
            // Relative cd targets are expanded before tracking: a target that
            // expands to a non-temp location fails closed.
            ("cd /tmp && cd $TMPDIR && touch f", true),
            ("cd /tmp && cd $HOME && touch f", false),
            ("cd /tmp && cd ~ && touch f", false),
            ("cd /tmp && cd - && touch f", false),
            ("cd /tmp && cd $OLDPWD && rm -rf x", false),
            ("cd /tmp && cd $OLDPWD && touch f", false),
            ("cd /tmp && cd $FOO && touch f", false),
            // $PWD after tracked cd resolves to the tracked CWD.
            ("cd /tmp && touch $PWD/out", true),
            ("cd /tmp && touch $PWD/../../etc/passwd", false),
            // A nonexistent relative target must not be tracked: `$PWD` would
            // then resolve one level deeper than the real CWD.
            ("cd /tmp ; cd src ; touch $PWD/../etc/passwd", false),
            (
                "mkdir -p /tmp/src && cd /tmp && cd src && touch $PWD/out",
                true,
            ),
            // cd to workspace + rm — blocked.
            ("cd / && rm f", false),
            // Relative path args without cd resolve to the workspace — blocked.
            ("touch f", false),
            // Relative globs stay rejected even under a temp CWD.
            ("cd /tmp && touch *.log", false),
            // `..`-normalized escape from a temp CWD — blocked.
            ("cd /tmp && echo hi > ../etc/passwd", false),
            // cp destination "." = current dir (workspace → blocked, temp → allowed).
            ("cp /tmp/a .", false),
            ("cd /tmp && cp /tmp/a .", true),
            // cd option flags are skipped before resolving the target.
            ("cd /tmp && cd -P /tmp && touch f", true),
            ("cd /tmp && cd -L /tmp && echo hi > out.txt", true),
            ("cd /tmp && cd -L -- /tmp && touch f", true),
            ("mkdir -p /tmp/snap && cd -P /tmp/snap && touch f", true),
            ("cd /tmp && cd -P /tmp/nonexistent_x && touch f", false),
            // Prefixed cd forms (`command`/`builtin`/`time`/`eval`) run the cd
            // in the current shell — routed and tracked like a bare cd.
            ("cd /tmp && command cd /tmp && touch f", true),
            ("cd /tmp && builtin cd /tmp && touch f", true),
            (
                "mkdir -p /tmp/snap && cd /tmp && time cd /tmp/snap && touch f",
                true,
            ),
            ("cd /tmp && eval cd /tmp && echo hi > out.txt", true),
            ("cd /tmp && command cd -L -- /tmp && touch f", true),
        ];

        run_cases(&cases);
    }

    #[test]
    fn cd_flag_forms_fail_closed() {
        let cases = [
            // Option flags were parsed as the cd target (tracking `/tmp/-P`
            // etc.) while the runtime cd lands in /etc, $HOME, or $OLDPWD —
            // the chained write escapes every temp root.
            ("cd /tmp && cd -P /etc && touch f", false),
            ("cd /tmp && cd -L /etc && rm -rf x", false),
            ("cd /tmp && cd -PL /etc && touch f", false),
            ("cd /tmp && cd -P && touch f", false),
            ("cd /tmp && cd -L && touch f", false),
            ("cd /tmp && cd -PL && touch f", false),
            ("cd /tmp && cd -- $HOME && touch f", false),
            ("cd /tmp && cd -- - && touch f", false),
            ("cd /tmp && cd -- - && rm -rf x", false),
            ("cd /tmp && cd -P - && touch f", false),
            // Invalid options error at runtime (the cd never happens), so
            // tracking them would approve a write that lands in the real CWD.
            ("cd /tmp && cd -e /tmp && touch f", false),
            ("cd /tmp && cd -eP /tmp && touch f", false),
            ("cd /tmp && cd -Pe /tmp && touch f", false),
            ("cd /tmp && cd -@ /tmp && touch f", false),
            ("cd /tmp && cd -x /tmp && touch f", false),
            ("cd /tmp && cd -P-L /tmp && touch f", false),
        ];

        run_cases(&cases);
    }

    // ── Phase 3 acceptance: git read-only operations ─────────────────────

    #[test]
    fn git_read_only_acceptance() {
        let cases = [
            // Allowed read forms (phase 3, decisions 3/4).
            ("git stash show", true),
            ("git stash show stash@{0}", true),
            ("git show-ref", true),
            ("git ls-remote", true),
            ("git submodule status", true),
            ("git submodule status --recursive", true),
            ("git submodule", true),
            ("git config user.name", true),
            ("git config --global user.name", true),
            ("git config --list", true),
            ("git config -l", true),
            ("git config --get user.name", true),
            ("git config --get-all core.pager", true),
            ("git config --name-only --get-regexp '^core\\.'", true),
            ("git rebase --show-current", true),
            ("git push --dry-run", true),
            ("git push -n", true),
            ("git push -n origin main", true),
            ("git clean -n", true),
            ("git clean --dry-run", true),
            ("git clean -ndx", true),
            ("git --version 2>&1", true),
            ("git status 2>&1", true),
            // Mutating siblings stay blocked.
            ("git config user.name Egor", false),
            ("git config --global user.name Egor", false),
            ("git config --add core.pager less", false),
            ("git config --edit", false),
            ("git config --unset user.name", false),
            ("git rebase --continue", false),
            ("git rebase main", false),
            ("git rebase", false),
            ("git push origin main", false),
            ("git push", false),
            ("git push --dry-run -f", false),
            ("git push -fn", false),
            ("git clean -f", false),
            ("git clean -n -f", false),
            ("git clean -fd", false),
            ("git clean", false),
            ("git stash pop", false),
            ("git stash apply", false),
            ("git stash drop", false),
            ("git stash push", false),
            ("git stash", false),
            // Exact stash subcommand matching: a mutation whose message
            // contains a read-form word must stay blocked (QA round: the
            // substring check allowed `git stash push -m "stash show"`).
            ("git stash push -m \"stash show\"", false),
            ("git stash push -m \"stash list\"", false),
            ("git stash store 0123abcd \"stash show\"", false),
            ("git stash show --stat stash@{0}", true),
            ("git stash list --oneline", true),
            ("git submodule foreach", false),
            ("git submodule update", false),
            ("git submodule add https://example.com/repo.git", false),
        ];

        run_cases(&cases);
    }

    // ── Phase 3 acceptance: cargo read-only invocations ──────────────────

    #[test]
    fn cargo_read_only_acceptance() {
        let cases = [
            // Toolchain specifier no longer becomes the parsed subcommand.
            ("cargo +nightly build", true),
            ("cargo +stable check", true),
            ("cargo +nightly test", true),
            ("cargo +nightly clippy", true),
            ("cargo +nightly doc", true),
            ("cargo +nightly run", false),
            ("cargo +nightly fix", false),
            ("cargo +nightly update", false),
            // stderr-capture suffix no longer becomes the parsed subcommand.
            ("cargo --version 2>&1", true),
            ("cargo build 2>&1", true),
            ("git --version 2>&1", true),
            // Help/version exemption for any subcommand, before `--`.
            ("cargo fix --help", true),
            ("cargo run --help", true),
            ("cargo run -h", true),
            ("cargo nextest --version", true),
            ("cargo --version", true),
            ("cargo build -V", true),
            // update --dry-run allowed; plain update blocked.
            ("cargo update --dry-run", true),
            ("cargo update", false),
            // Still blocked.
            ("cargo fix", false),
            ("cargo run", false),
            ("cargo run -- --help", false),
            ("cargo clippy --fix", false),
        ];

        run_cases(&cases);
    }

    // ── Phase 3 acceptance: keep-blocked battery ─────────────────────────

    #[test]
    fn keep_blocked_battery() {
        let cases = [
            // Process control — all forms.
            ("kill 1234", false),
            ("kill -9 1234", false),
            ("pkill -f mahbot", false),
            ("killall chrome", false),
            // ln in all forms.
            ("ln -s /tmp/a /tmp/b", false),
            ("ln /tmp/a /tmp/b", false),
            // unzip extract.
            ("unzip archive.zip", false),
            ("unzip -o archive.zip -d /tmp/out", false),
            // Live-DB sidecar deletion (tilde / $HOME spellings).
            ("rm ~/.mahbot/db/board.db-wal", false),
            ("rm $HOME/.mahbot/db/board.db-wal", false),
            // rm -rf target.
            ("rm -rf target", false),
            // cargo mutators.
            ("cargo clippy --fix", false),
            ("cargo run", false),
            ("cargo fix", false),
            // git init even in temp.
            ("git init", false),
            ("git init /tmp/x", false),
            // Non-read-only git mutations.
            ("git commit -m test", false),
            ("git reset --hard", false),
            // sed/awk/dd/curl/wget on workspace targets.
            ("sed -i 's/a/b/' file", false),
            ("awk -i inplace '{print $1}' file", false),
            ("dd if=/dev/zero of=file bs=1 count=1", false),
            ("curl -o out.txt URL", false),
            ("wget -O out.txt URL", false),
        ];

        run_cases(&cases);
    }

    // ── Temp-value variable binding ──────────────────────────────────────

    /// `NAME=$(mktemp -d)` / backtick spellings bind a fresh temp root with an
    /// unknown concrete value; writes via `$NAME` resolve under that root.
    /// Non-temp and unknown bindings poison the variable (fail-closed);
    /// temp re-binds clear the poison.
    #[test]
    fn temp_var_binding_acceptance() {
        let cases = [
            // mktemp substitution binds a temp root; writes via $SNAP allowed.
            ("SNAP=$(mktemp -d)\ntouch \"$SNAP/f\"", true),
            ("SNAP=$(mktemp -d)\necho hi > \"$SNAP/out\"", true),
            ("SNAP=$(mktemp -d)\ncp /etc/passwd \"$SNAP/out\"", true),
            ("SNAP=$(mktemp -d)\nmkdir -p \"$SNAP/a/b\"", true),
            ("SNAP=$(mktemp -d)\nrm -rf \"$SNAP\"", true),
            ("SNAP=$(mktemp -d)\nrm -rf \"$SNAP/\"", true),
            // Backtick and quoted-substitution spellings.
            ("SNAP=`mktemp -d`\ntouch \"$SNAP/f\"", true),
            ("SNAP=\"$(mktemp -d)\"\ntouch \"$SNAP/f\"", true),
            // export form.
            ("export SNAP=$(mktemp -d)\ntouch \"$SNAP/f\"", true),
            // Binding inside a substitution applies within the subshell.
            ("echo $(SNAP=$(mktemp -d); touch \"$SNAP/f\")", true),
            // Braced / unbraced references.
            ("SNAP=$(mktemp -d)\ntouch $SNAP/f", true),
            ("SNAP=$(mktemp -d)\ntouch ${SNAP}/f", true),
            // Concrete temp bindings track too.
            ("SNAP=/tmp/snap\ntouch \"$SNAP/f\"", true),
            ("SNAP=$TMPDIR/snap\ntouch \"$SNAP/f\"", true),
            (
                "SNAP=/tmp/snap\ndir=$SNAP/.mahbot/db\nmkdir -p \"$dir\"",
                true,
            ),
            // Non-temp / unknown bindings poison → blocked.
            ("SNAP=/etc\ntouch \"$SNAP/f\"", false),
            (
                "SNAP=/__mahbot_readonly_test_ws__/snap\nmkdir -p \"$SNAP/db\"",
                false,
            ),
            ("SNAP=$FOO\ntouch \"$SNAP/f\"", false),
            // Temp rebind clears; non-temp rebind stays poisoned.
            ("SNAP=/etc\nSNAP=$(mktemp -d)\ntouch \"$SNAP/f\"", true),
            ("SNAP=$(mktemp -d)\nSNAP=/etc\ntouch \"$SNAP/f\"", false),
            (
                "SNAP=$(mktemp -d)\nSNAP=/tmp/other\ntouch \"$SNAP/f\"",
                true,
            ),
            // Env-prefix form.
            ("SNAP=/tmp/snap touch $SNAP/f", true),
            ("SNAP=/etc touch $SNAP/f", false),
            // Unbound variable without a temp anchor stays blocked.
            ("touch $FOO/f", false),
            // mktemp with a template still binds.
            (
                "SNAP=$(mktemp -d /tmp/mahbot.XXXXXX)\ntouch \"$SNAP/f\"",
                true,
            ),
            // Positional templates that provably resolve under a temp root
            // bind — including the `-- <template>` form and variables
            // expanding to a temp path.
            (
                "SNAP=$(mktemp -d -- /tmp/mahbot.XXXXXX)\ntouch \"$SNAP/f\"",
                true,
            ),
            (
                "SNAP=$(mktemp -d \"$TMPDIR/mahbot.XXXXXX\")\ntouch \"$SNAP/f\"",
                true,
            ),
            // Absolute non-temp, $HOME, relative (shell-cwd) and workspace
            // templates fail closed — no temp-root binding, writes via the
            // variable reject.
            (
                "SNAP=$(mktemp -d /etc/mahbot.XXXXXX)\ntouch \"$SNAP/f\"",
                false,
            ),
            (
                "SNAP=$(mktemp -d /__mahbot_readonly_test_ws__/snap.XXXXXX)\ntouch \"$SNAP/f\"",
                false,
            ),
            (
                "SNAP=$(mktemp -d $HOME/snap.XXXXXX)\ntouch \"$SNAP/f\"",
                false,
            ),
            ("SNAP=$(mktemp -d snap.XXXXXX)\ntouch \"$SNAP/f\"", false),
            ("SNAP=$(mktemp -d ./snap.XXXXXX)\ntouch \"$SNAP/f\"", false),
            // The `-- <template>` form carries the same rules.
            (
                "SNAP=$(mktemp -d -- /etc/mahbot.XXXXXX)\ntouch \"$SNAP/f\"",
                false,
            ),
            ("SNAP=$(mktemp -d -- snap.XXXXXX)\ntouch \"$SNAP/f\"", false),
            (
                "SNAP=$(mktemp -d -- $HOME/snap.XXXXXX)\ntouch \"$SNAP/f\"",
                false,
            ),
            // A second positional template makes mktemp error at runtime —
            // the variable would bind empty, so no temp anchor.
            (
                "SNAP=$(mktemp -d /tmp/a.XXXXXX /tmp/b.XXXXXX)\ntouch \"$SNAP/f\"",
                false,
            ),
            // A template without trailing X's makes mktemp error at runtime —
            // the variable would bind empty, so no temp anchor.
            ("SNAP=$(mktemp -d /tmp/foo)\ntouch \"$SNAP/f\"", false),
            ("SNAP=$(mktemp -d /tmp/foo.XX)\ntouch \"$SNAP/f\"", false),
            // mktemp -p/--tmpdir to a provably existing temp dir binds; to a
            // non-temp dir does not. mktemp never creates the -p directory,
            // so a nonexistent one makes it error at runtime and bind the
            // variable empty — the chained write would land outside every
            // temp root (false temp anchor), so it fails closed.
            ("SNAP=$(mktemp -d -p /tmp)\ntouch \"$SNAP/f\"", true),
            ("SNAP=$(mktemp -d --tmpdir=/tmp)\ntouch \"$SNAP/f\"", true),
            ("SNAP=$(mktemp -d -p /etc)\ntouch \"$SNAP/f\"", false),
            (
                "SNAP=$(mktemp -d --tmpdir=/__mahbot_readonly_test_ws__)\ntouch \"$SNAP/f\"",
                false,
            ),
            (
                "SNAP=$(mktemp -d -p /tmp/nonexistent_x)\ntouch \"$SNAP/f\"",
                false,
            ),
            (
                "SNAP=$(mktemp -d -p /tmp/nonexistent_x) ; touch \"$SNAP/f\"",
                false,
            ),
            (
                "SNAP=$(mktemp -d --tmpdir=/tmp/nonexistent_x)\ntouch \"$SNAP/f\"",
                false,
            ),
            // A -p dir created by `mkdir` earlier in the chain is provable.
            (
                "mkdir -p /tmp/snapdir; SNAP=$(mktemp -d -p /tmp/snapdir)\ntouch \"$SNAP/f\"",
                true,
            ),
            // The space-separated `--tmpdir DIR` form is not provable (GNU
            // optional-arg semantics error there; macOS joins DIR as a
            // template under /tmp) → fail-closed.
            ("SNAP=$(mktemp -d --tmpdir /tmp)\ntouch \"$SNAP/f\"", false),
            ("SNAP=$(mktemp -d --tmpdir /etc)\ntouch \"$SNAP/f\"", false),
            // `--suffix=` is GNU-only; macOS mktemp rejects it → fail-closed.
            (
                "SNAP=$(mktemp -d --suffix=.foo /tmp/x.XXXXXX)\ntouch \"$SNAP/f\"",
                false,
            ),
            // The same fail-closed shapes must hold when the initial cwd is a
            // tracked temp dir: the raw `$(mktemp ...)` substitution string
            // must never resolve as a literal path against a tracked temp cwd
            // and bind as a temp value — at runtime mktemp errors, SNAP binds
            // empty, and the chained write lands outside every temp root.
            (
                "cd /tmp; SNAP=$(mktemp -d -p /tmp/nonexistent_x); touch $SNAP/f",
                false,
            ),
            (
                "cd /tmp\nSNAP=$(mktemp -d -p /tmp/nonexistent_x)\ntouch $SNAP/f",
                false,
            ),
            (
                "cd /tmp && SNAP=$(mktemp -d -p /tmp/nonexistent_x) && touch $SNAP/f",
                false,
            ),
            ("cd /tmp; SNAP=$(mktemp -d -p /etc); touch $SNAP/f", false),
            (
                "cd /tmp; SNAP=$(mktemp -d /etc/foo.XXXXXX); touch $SNAP/f",
                false,
            ),
            (
                "cd /tmp; SNAP=$(mktemp -d --tmpdir /tmp); touch $SNAP/f",
                false,
            ),
            (
                "cd /tmp; SNAP=$(mktemp -d --tmpdir=/tmp/nonexistent_x); touch $SNAP/f",
                false,
            ),
            (
                "cd /tmp; SNAP=$(mktemp -d --suffix=.foo /tmp/x.XXXXXX); touch $SNAP/f",
                false,
            ),
            // Workspace-targeting chains through the unprovable substitution
            // stay blocked.
            (
                "cd /tmp; SNAP=$(mktemp -d -p /etc); rm -rf $SNAP/Users/egordezic/Desktop/mahbot",
                false,
            ),
            (
                "cd /tmp; SNAP=$(mktemp -d -p /etc); echo pwn > $SNAP/Users/egordezic/Desktop/mahbot/x",
                false,
            ),
            (
                "cd /tmp; SNAP=$(mktemp -d -p /etc); tee $SNAP/Users/egordezic/Desktop/mahbot/x",
                false,
            ),
            // A substitution used directly in a path argument is equally
            // unprovable: at runtime mktemp errors, the empty expansion makes
            // the write land at /f or the workspace path.
            ("cd /tmp; touch $(mktemp -d -p /etc)/f", false),
            ("cd /tmp; echo hi > $(mktemp -d -p /etc)/f", false),
            (
                "cd /tmp; rm -rf $(mktemp -d -p /etc)/Users/egordezic/Desktop/mahbot",
                false,
            ),
            ("cd /tmp; touch `mktemp -d -p /etc`/f", false),
            // Provable bindings keep working under a tracked temp cwd.
            ("cd /tmp; SNAP=$(mktemp -d); touch $SNAP/f", true),
            ("cd /tmp; SNAP=$(mktemp -d -p /tmp); touch $SNAP/f", true),
            (
                "cd /tmp; SNAP=$(mktemp -d --tmpdir=/tmp); touch $SNAP/f",
                true,
            ),
            // A single-quoted `$(...)` is a literal, not a substitution: it
            // never binds a temp root. A substitution embedded in a larger
            // value is unprovable too — fail-closed.
            ("SNAP='$(mktemp -d)'\ntouch \"$SNAP/f\"", false),
            (
                "cd /tmp; SNAP=pre$(mktemp -d -p /etc); touch $SNAP/f",
                false,
            ),
            // Parameter expansions with modifiers (`${FOO:-../etc}`) are
            // unprovable: the default can carry `..` and escape the temp
            // root at runtime, so words containing them fail to resolve.
            ("cd /tmp; echo hi > /tmp/${FOO:-../etc}/f", false),
            ("echo hi > /tmp/${FOO:-../etc}/f", false),
            ("cd /tmp; touch /tmp/${FOO//x/../etc}/f", false),
            ("cd /tmp; echo hi > \"/tmp/${FOO:-../etc}/f\"", false),
        ];

        run_cases(&cases);
    }

    /// Opaque suffixes (`$$`, `$RANDOM`) under a temp prefix resolve; without
    /// a temp anchor, or after an escape segment, they stay fail-closed. An
    /// arbitrary unbound variable is never treated as an opaque suffix —
    /// only the `$RANDOM` builtin is.
    #[test]
    fn opaque_suffix_acceptance() {
        let cases = [
            ("SNAP=$(mktemp -d)\ntouch \"$SNAP/$RANDOM/f\"", true),
            ("SNAP=$(mktemp -d)\ntouch \"$SNAP/$$/f\"", true),
            ("SNAP=$(mktemp -d)\ntouch \"$SNAP/$RANDOM\"", true),
            ("SNAP=$(mktemp -d)\necho hi > \"$SNAP/$RANDOM/out\"", true),
            ("touch \"/tmp/$RANDOM/f\"", true),
            // Without a temp anchor the opaque variable is unprovable.
            ("touch \"$RANDOM/f\"", false),
            // Arbitrary unbound variables fail closed even after a temp anchor.
            ("SNAP=$(mktemp -d)\ntouch \"$SNAP/$FOO/f\"", false),
            ("SNAP=$(mktemp -d)\ntouch \"$SNAP/$PATH/f\"", false),
            // `..` after an opaque suffix is unverifiable → blocked.
            ("SNAP=$(mktemp -d)\ntouch \"$SNAP/$RANDOM/../f\"", false),
            (
                "SNAP=$(mktemp -d)\ntouch \"$SNAP/$RANDOM/../../etc/passwd\"",
                false,
            ),
            // `..` before the opaque suffix normalizes concretely → allowed.
            ("SNAP=$(mktemp -d)\ntouch \"$SNAP/../$RANDOM/f\"", true),
        ];

        run_cases(&cases);
    }

    /// The documented snapshot-query procedure (idiomatic and literal
    /// spellings) passes; workspace-targeting variants of the same shapes stay
    /// blocked.
    #[test]
    fn snapshot_query_procedure_acceptance() {
        let cases = [
            // Idiomatic spelling.
            (
                "SNAP=$(mktemp -d)\nmkdir -p \"$SNAP/.mahbot/db\"\ncp ~/.mahbot/db/board.db \"$SNAP/.mahbot/db/\"\nHOME=\"$SNAP\" mahbot debug --db board \"SELECT 1\"\nrm -rf \"$SNAP\"",
                true,
            ),
            // Full for-loop spelling from the deleted ops doc.
            (
                "SNAP=$(mktemp -d)\nmkdir -p \"$SNAP/.mahbot/db\"\nfor db in ~/.mahbot/db/*.db; do\ncp \"$db\" \"$SNAP/.mahbot/db/\"\ncp \"$db-wal\" \"$SNAP/.mahbot/db/\" 2>/dev/null || true\ndone\nHOME=\"$SNAP\" mahbot debug --db sessions \"SELECT COUNT(*) FROM messages\"\nrm -rf \"$SNAP\"",
                true,
            ),
            // Literal spelling: concrete temp bindings.
            (
                "SNAP=/tmp/mahbot-snap\ndir=$SNAP/.mahbot/db\nmkdir -p \"$dir\"\ncp ~/.mahbot/db/board.db \"$dir/\"\nHOME=\"$SNAP\" mahbot debug --db board \"SELECT 1\"\nrm -rf \"$SNAP\"",
                true,
            ),
            // Workspace-targeting variants of the same shapes stay blocked.
            (
                "SNAP=/__mahbot_readonly_test_ws__/snap\nmkdir -p \"$SNAP/.mahbot/db\"",
                false,
            ),
            (
                "SNAP=$(mktemp -d)\ncp ~/.mahbot/db/board.db /__mahbot_readonly_test_ws__/out",
                false,
            ),
            (
                "SNAP=$(mktemp -d)\nrm -rf /__mahbot_readonly_test_ws__",
                false,
            ),
        ];

        run_cases(&cases);
    }

    /// Denial messages teach the recognized spelling: a literal temp path or
    /// `NAME=$(mktemp -d)` for unresolvable variables, and `cargo build` +
    /// direct binary invocation for `cargo run`.
    #[test]
    fn denial_message_education() {
        let ctx = test_ctx();
        let err = check_command("touch $FOO/out", &ctx).unwrap_err();
        assert!(
            err.contains("$(mktemp -d)"),
            "scratch-mutator denial should teach the variable spelling: {err}"
        );
        let err = check_command("rm $FOO/out", &ctx).unwrap_err();
        assert!(
            err.contains("$(mktemp -d)"),
            "temp-mutator denial should teach the variable spelling: {err}"
        );
        let err = check_command("cargo run", &ctx).unwrap_err();
        assert!(
            err.contains("cargo build") && err.contains("target/debug"),
            "cargo run denial should teach the built-binary alternative: {err}"
        );
    }

    #[test]
    fn symlink_escape_fails_closed() {
        // A symlink inside a temp root pointing at an existing non-temp dir:
        // writes through it land outside every temp root. The target must
        // exist for canonicalize to resolve through it; the project directory
        // is a stable non-temp anchor.
        let target = std::env::current_dir().expect("cwd");
        let link = std::env::temp_dir().join(format!("mahbot_ro_probe_{}", std::process::id()));
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&target, &link).expect("symlink");
        let link_str = link.to_string_lossy().into_owned();
        let cases = [
            (format!("cd {link_str} && touch f"), false),
            (format!("cd {link_str} ; touch f"), false),
            (format!("touch {link_str}/f"), false),
            (format!("rm -rf {link_str}/x"), false),
            (format!("echo hi > {link_str}/out.txt"), false),
            (format!("tee {link_str}/x"), false),
            (format!("cp /tmp/a {link_str}/dest"), false),
        ];
        for (command, allowed) in &cases {
            if *allowed {
                ok(command);
            } else {
                assert_rejected(command);
            }
        }
        let _ = std::fs::remove_file(&link);
    }

    #[test]
    fn prefixed_cd_and_dotdot_escapes_fail_closed() {
        let cases = [
            // Prefixed cd forms (`command`/`builtin`/`time`/`eval`) run the cd
            // in the current shell; a non-temp target must reset fail-closed.
            ("cd /tmp && command cd /etc && touch f", false),
            ("cd /tmp && command cd -- /etc && touch f", false),
            ("cd /tmp && builtin cd $HOME && touch f", false),
            ("cd /tmp && builtin cd -P /etc && touch f", false),
            ("cd /tmp && time cd ~ && touch f", false),
            ("cd /tmp && eval cd \"$HOME\" && touch f", false),
            (
                "cd /tmp && command cd /__mahbot_readonly_test_ws__ && rm -rf x",
                false,
            ),
            (
                "cd /tmp && builtin cd /__mahbot_readonly_test_ws__ && touch f",
                false,
            ),
            // `..`-collapsed missing-prefix targets (mkdir-created tail).
            (
                "mkdir -p /tmp/qa_b && cd /tmp/qa_a/../qa_b && touch ../x",
                false,
            ),
            (
                "mkdir -p /tmp/qa_b && cd /tmp && cd qa_a/../qa_b && touch ../x",
                false,
            ),
            (
                "mkdir -p /tmp/qa_b && cd /tmp && SNAP=$(mktemp -d -p qa_a/../qa_b) && touch $SNAP/f",
                false,
            ),
            (
                "mkdir -p /tmp/qa_b && cd /tmp && SNAP=$(mktemp -d qa_a/../qa_b.XXXXXX) && touch $SNAP/f",
                false,
            ),
            // The `..`-collapsed shape via the -P flag path.
            (
                "mkdir -p /tmp/qa_b && cd /tmp && cd -P qa_a/../qa_b && touch ../x",
                false,
            ),
            // Prefixed cd to a temp target stays accepted (routed, tracks).
            ("cd /tmp && command cd /tmp && touch f", true),
            ("cd /tmp && builtin cd -P /tmp && touch f", true),
        ];

        run_cases(&cases);
    }
}
