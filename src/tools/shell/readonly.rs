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
    "hash-object",
    "stripspace",
    "remote",
    "branch",
    "tag",
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
        // After a heredoc marker, the rest of the delimiter line stays scanned
        // (a redirect there is a real write target) and may declare further
        // heredocs; once the line ends, switch to body-skipping. The newline
        // branch is queue-conditional because only a pending heredoc has a
        // delimiter-line tail to end.
        if !queue.is_empty() && chars[i].1 == '\n' {
            out.push('\n');
            i += 1;
            skipping_body = true;
            continue;
        }

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
                None => break, // dangling `<<` at end of input
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
/// Shared by the scanner's single dispatch chain so the marker grammar (fd
/// prefix, `<<-` tabs, whitespace, quoted delimiter) lives in exactly one
/// place.
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
    if i >= chars.len() {
        return None; // dangling `<<` at end of input — nothing to strip
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
        if let Some((_, next)) = substitution_span(line, j) {
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
        if !escaped
            && !in_single
            && let Some((_, next)) = substitution_span(segment, i)
        {
            i = next;
            continue;
        }
        // Bare `((...))` arithmetic span: `>`/`<` inside are comparisons, not
        // redirects (`(( i > 3 ))`, `for ((i=0; i>3; i++))` — false-rejected
        // before this skip). Adjacent parens are arithmetic; `( (` with a
        // space is a nested subshell and stays scanned. Skipping inside
        // quotes would risk swallowing a real redirect after an unterminated
        // quoted `((`, so only unquoted spans are skipped.
        if !escaped && !in_single && !in_double && c == '(' && bytes.get(i + 1) == Some(&b'(') {
            i = find_arithmetic_end(segment, i);
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
            return Err(disallowed_redirect_err(segment));
        }

        // Allowed targets
        if target == "/dev/null" {
            continue;
        }

        if writes_outside_temp(target, state) {
            // Absolute/relative non-temp non-devnull = disallowed
            return Err(disallowed_redirect_err(segment));
        }
    }

    Ok(())
}

/// Rejection message for any disallowed output redirect (bare or non-temp).
fn disallowed_redirect_err(segment: &str) -> String {
    format!(
        "⚠️ Read-only mode: command contains a disallowed output redirect.\n\
         Command: `{segment}`\n\
         Redirects are only allowed to /dev/null, 2>&1, 1>&2, or paths under /tmp, /var/tmp, or the OS temp directory.\n\
         Suggestion: pipe to a pager (e.g., `| less`) or use `| head` to limit output."
    )
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
    // temp root, so they always reject.
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
        if crate::tools::path::contains_glob(&expanded, false) {
            return None;
        }
        let cwd = state.cwd.as_ref()?; // fail-closed when tracking was reset
        return Some(cwd.join(&expanded));
    }
    Some(path)
}

/// True when `word` does not provably resolve under an allowed temp root.
/// Resolution failure counts as outside temp (fail-closed): unknown or
/// poisoned variables, tilde paths, and unanchored relative paths all reject.
fn writes_outside_temp(word: &str, state: &ValidationState) -> bool {
    !resolve_path_word(word, state).is_some_and(|p| is_path_under_temp(&p, state.ctx))
}

/// Strip balanced surrounding single/double quotes from a shell word.
/// Returns `(content, was_single_quoted)`, or `None` for unbalanced/mixed
/// quotes (e.g. `"/tmp`, `'/tmp"`, `ab"cd`).
pub(super) fn strip_outer_quotes(word: &str) -> Option<(&str, bool)> {
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
            // Word-final backslash cannot resolve to a concrete path —
            // fail closed instead of panicking.
            let c = next.chars().next()?;
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
///   when tracking was reset (fail-closed). An explicit `PWD=...` assignment
///   in the chain takes precedence over tracked-cwd resolution (bash sets
///   `$PWD` from the assignment, and a poisoned non-temp assignment must not
///   be masked by the tracked CWD).
/// - `$HOME` and poisoned variables are always `Blocked`.
/// - Bound variables (TMPDIR/TMP/TEMP from the session env, plus any variable
///   assigned a temp-root value in the chain) resolve to their value or
///   [`VarValue::TempRoot`] when the value is a fresh `$(mktemp -d)` root.
/// - Unbound variables are [`VarValue::Opaque`].
fn resolve_var(name: &str, state: &ValidationState) -> VarValue {
    match name {
        "PWD" => match state.vars.get(name) {
            // Explicit assignment wins: `PWD=/etc` must not be masked by the
            // tracked CWD (`cd /tmp && PWD=/etc touch $PWD/f` → /etc/f).
            Some(b) => {
                if b.poisoned {
                    VarValue::Blocked
                } else {
                    match &b.value {
                        Some(v) => VarValue::Concrete(v.clone()),
                        None => VarValue::TempRoot,
                    }
                }
            }
            None => match &state.cwd {
                Some(cwd) => VarValue::Concrete(cwd.to_string_lossy().into_owned()),
                None => VarValue::Blocked,
            },
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
/// contract).
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
    /// shell environment. The daemon's shell launcher sets `TMPDIR` from
    /// [`super::TMPDIR_BASELINE`] (see `baseline_env_value`), so the
    /// production default is exactly that; tests inject their own.
    temp_vars: Vec<(String, String)>,
}

impl CheckContext {
    /// Context for a session rooted at `workspace_root` with the standard
    /// OS temp roots and the session shell's temp-variable bindings.
    pub(super) fn for_workspace(workspace_root: &Path) -> Self {
        Self {
            workspace_root: workspace_root.to_path_buf(),
            temp_roots: crate::tools::path::allowed_temp_roots(),
            temp_vars: vec![("TMPDIR".to_string(), super::TMPDIR_BASELINE.to_string())],
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
    /// (any non-literal-absolute `cd` form resets tracking).
    cwd: Option<std::path::PathBuf>,
    /// Number of `cd`/`pushd`/`popd` verbs (incl. eval-body cds) executed so
    /// far in the current-shell context. Construct parsers compare a part's
    /// count against the construct start to detect a cd that leaks into the
    /// parent shell (if/for/while/case/select bodies run in the current
    /// shell): the leaked CWD is untrackable, so the outer tracking resets.
    cd_count: u64,
    /// Tracked variable bindings by name: the session temp variables
    /// (TMPDIR/TMP/TEMP) plus any variable assigned a temp-root value
    /// (`SNAP=$(mktemp -d)`, `SNAP=/tmp/x`, ...) in the command chain.
    vars: std::collections::HashMap<String, VarBinding>,
    /// Directories provably created by an approved `mkdir` earlier in this
    /// command chain (normalized temp paths). A later `cd` into one of them
    /// tracks even when the directory did not exist at validation time.
    created_dirs: std::collections::HashSet<std::path::PathBuf>,
}

impl<'a> ValidationState<'a> {
    /// Initial state: CWD = the session's workspace root; temp vars bound from
    /// the context's session-environment snapshot.
    fn new(ctx: &'a CheckContext) -> ValidationState<'a> {
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
            cd_count: 0,
            vars,
            created_dirs: std::collections::HashSet::new(),
        }
    }

    /// Snapshot of the current tracking state. Used for command substitutions,
    /// which execute in a subshell: `cd`/export inside `$(...)` must not leak
    /// to the outer command's tracking. Returns a state with the same context
    /// lifetime (not tied to `&self`), so snapshots can outlive the borrow.
    fn snapshot(&self) -> ValidationState<'a> {
        ValidationState {
            ctx: self.ctx,
            cwd: self.cwd.clone(),
            cd_count: self.cd_count,
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

/// Validate a command string (which may contain heredocs, substitutions,
/// pipelines, background jobs, and compound constructs) against `state`.
///
/// Heredoc bodies are stripped before any scanning (body text is literal —
/// never commands, redirects, or substitutions; substitutions inside
/// *unquoted* bodies are emitted by the stripper so they remain scanned).
/// The result is walked by [`scan_list`], which validates each simple command
/// and recurses into compound constructs (if/for/while/until/select/case,
/// brace groups, subshells, function definitions).
fn validate_string(s: &str, state: &mut ValidationState) -> Result<(), String> {
    let scan = strip_heredoc_bodies(s);
    scan_list(&scan, 0, &[], &[], state, false)?;
    Ok(())
}

/// Validate a single simple command (no compound constructs) against `state`:
/// redirect scan, substitution scan, then the segment check. Used by
/// [`scan_list`] for both chain members and pipeline members (the latter get
/// a fresh snapshot so their state changes never leak). `negated` marks a
/// `!`-prefixed segment (`time` demoted to external there).
fn validate_simple_command(
    seg: &str,
    state: &mut ValidationState,
    negated: bool,
) -> Result<(), String> {
    let seg = seg.trim();
    if seg.is_empty() {
        return Ok(());
    }
    // Output redirects are validated against the state BEFORE this segment's
    // own `cd`/export bindings: bash expands redirect targets with the shell's
    // current variables, and an env-prefix assignment does not affect them.
    has_disallowed_redirect(seg, state)?;
    // Command substitutions are validated with the state as of this segment —
    // prior segments' `cd`/export bindings apply. Like redirects, a
    // substitution does not see its own segment's env-prefix (it expands
    // before the prefix is applied to the command), so substitution scanning
    // runs before `check_segment` applies this segment's bindings. Nested
    // validation snapshots the state: `cd`/export inside `$(...)` runs in a
    // subshell and must not leak to the outer command.
    scan_substitutions(seg, state)?;
    check_segment(seg, state, negated)
}

// ── Construct-aware list walker ────────────────────────────────────────

/// True when `s` (trailing whitespace trimmed) ends with an unescaped `>` or
/// `<` — an `&`/`|` immediately following is a redirect operator (`2>&1`,
/// `>|`, …), not a separator. A trailing backslash escapes the metachar
/// (`echo a\> & touch f`: the `>` is a literal argument, so the `&` IS a
/// background separator).
///
/// Quote awareness is implicit: the caller's `current` buffer preserves quote
/// characters, so a quoted `>`/`<` is always followed by its closing quote
/// (`echo ">" &` → `current` ends with `"` → separator, matching bash where
/// the quoted `>` is a literal argument).
fn ends_with_unescaped_redirect_op(s: &str) -> bool {
    let s = s.trim_end();
    if !s.ends_with(['>', '<']) {
        return false;
    }
    let backslashes = s.chars().rev().skip(1).take_while(|c| *c == '\\').count();
    backslashes % 2 == 0
}

/// Walk a shell list (a top-level command string or a construct body) from
/// byte `start`, validating each simple command and recursing into compound
/// constructs, until a terminator appears or the input ends.
///
/// `stop_keywords` are matched at command position (word boundaries): the
/// construct keywords (`then`, `do`, `done`, `fi`, `elif`, `else`, `esac`)
/// plus the single-char `}`/`)` group terminators. `stop_tokens` are matched
/// at any unquoted position: `;;`, `;&`, `;;&` (case-body terminators, which
/// bash does not require to be preceded by whitespace).
///
/// Returns the index after the terminator (or `s.len()` at end of input).
///
/// State semantics: `&&`/`||`/`;`/newline thread the state between commands.
/// Pipeline members (`|`, `|&`) run in subshells: each member is validated
/// against a snapshot of the state at pipeline start and its changes never
/// leak. A single `&` backgrounds the whole preceding chain: the chain ran in
/// a child, so the state is restored to the chain start. Stray terminators at
/// command position reject (fail-closed).
///
/// A leading `!` negates the exit status of the whole following pipeline but
/// only the direct list: it demotes a head-of-list `time` to the external
/// command (bash recognizes the reserved word only at a fresh command start),
/// while `cd` and everything else still run in the current shell. The flag
/// resets at `&&`/`||`/`;`/newline/`&` (a new list) and is not passed into
/// nested constructs (`! { time cd /tmp; }` keeps `time` reserved inside the
/// brace body).
///
/// `time_external_head`: the FIRST validated unit of this list runs after a
/// keyword where bash does not recognize `time` as the reserved word — the
/// first word of an `if`/`elif`/`while`/`until` condition or of a case-arm
/// body (probe-verified: `if time cd /etc; ...` runs external `/usr/bin/time`
/// with the timed cd in a child). Only that unit demotes; a separator or
/// newline before the first word returns `time` to reserved (`if \n time cd
/// /etc` is the reserved word again).
#[allow(clippy::too_many_lines)] // security-critical list walker
fn scan_list(
    s: &str,
    start: usize,
    stop_keywords: &[&str],
    stop_tokens: &[&str],
    state: &mut ValidationState,
    time_external_head: bool,
) -> Result<usize, String> {
    let mut i = start;
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    // `!`-negation context: set when a `!` is consumed at command position,
    // cleared at list separators (the negation applies to one pipeline only).
    let mut negated = false;
    // Condition/case-arm head: demotes `time` for the first validated unit
    // only (see `time_external_head`); consumed at the first flush.
    let mut time_external_pending = time_external_head;
    // State at the start of the current chain — restored at `&` boundaries
    // (the whole chain ran in a background child).
    let mut chain_start = state.snapshot();
    // State at the start of the current pipeline — pipeline members validate
    // against snapshots of it; `state` itself is never mutated by members.
    let mut pipeline_state: Option<ValidationState> = None;
    let mut in_pipeline = false;

    while i < s.len() {
        let c = s[i..].chars().next().expect("i < s.len()");

        // Backslash: preserve the escape so downstream scans see it
        // (`c\144` is an unprovable verb, `\;` is an escaped separator).
        // Backslash-newline is a line continuation (joined, not a separator).
        if c == '\\' && !in_single {
            let Some(next) = s[i + c.len_utf8()..].chars().next() else {
                current.push(c);
                break;
            };
            if next == '\n' {
                i += c.len_utf8() + next.len_utf8();
                continue;
            }
            current.push(c);
            current.push(next);
            i += c.len_utf8() + next.len_utf8();
            continue;
        }

        // Command substitutions stay whole: separators inside `$(...)` are
        // part of the substitution, not list separators (their content is
        // validated by the substitution scan with subshell semantics).
        if !escaped
            && !in_single
            && let Some((_, next)) = substitution_span(s, i)
        {
            current.push_str(&s[i..next]);
            i = next;
            continue;
        }

        // Bare `((...))` arithmetic spans stay whole (their `;`/`)` are
        // arithmetic, not list structure).
        if !escaped
            && !in_single
            && !in_double
            && c == '('
            && s.as_bytes().get(i + 1) == Some(&b'(')
        {
            let next = find_arithmetic_end(s, i);
            current.push_str(&s[i..next]);
            i = next;
            continue;
        }

        if !super::track_char_context(c, &mut in_single, &mut in_double, &mut escaped) {
            current.push(c);
            i += c.len_utf8();
            continue;
        }

        // Case-body terminators (`;;`, `;&`, `;;&`) match at any position —
        // bash does not require a space before them (`echo hi;;`).
        if !in_single
            && !in_double
            && !escaped
            && let Some(tok) = stop_tokens.iter().find(|t| s[i..].starts_with(**t))
        {
            validate_unit(
                &current,
                state,
                pipeline_state.as_ref(),
                in_pipeline,
                negated || time_external_pending,
            )?;
            current.clear();
            return Ok(i + tok.len());
        }

        // At command position: compound constructs, terminators, `!` negation.
        // Only at non-whitespace chars — a whitespace position must not read
        // ahead past the whitespace into a following token (the ` }` of a
        // brace group must be caught by the stop-token check at the `}`, not
        // treated as a stray terminator from the space position).
        if current.trim().is_empty()
            && !c.is_whitespace()
            && let Some(keyword) = read_keyword_at(s, i)
        {
            if stop_keywords.contains(&keyword.as_str()) {
                validate_unit(
                    &current,
                    state,
                    pipeline_state.as_ref(),
                    in_pipeline,
                    negated || time_external_pending,
                )?;
                current.clear();
                return Ok(i + keyword.len());
            }
            // `!` pipeline negation — the following pipeline runs in the
            // current shell but `time` loses its reserved-word status (the
            // external command's timed command runs in a child). Skip the
            // operator; the negation applies until the next list separator.
            if keyword == "!" {
                negated = true;
                i += 1;
                continue;
            }
            // A pipeline-member construct runs in a subshell: validate it
            // against a snapshot of the pipeline-start state and discard the
            // result (its bindings/cwd must not leak into the parent). Chain
            // members use the threaded `state` (brace groups run in the
            // current shell).
            if in_pipeline {
                let mut snap = pipeline_state
                    .as_ref()
                    .expect("pipeline state set while in pipeline")
                    .snapshot();
                match handle_construct(s, i, &mut snap)? {
                    ConstructAction::Consumed(after) => {
                        i = after;
                        continue;
                    }
                    ConstructAction::Accumulate(after) => {
                        current.push_str(&s[i..after]);
                        i = after;
                        continue;
                    }
                    ConstructAction::NotConstruct => {}
                }
            } else {
                match handle_construct(s, i, state)? {
                    ConstructAction::Consumed(after) => {
                        i = after;
                        continue;
                    }
                    ConstructAction::Accumulate(after) => {
                        current.push_str(&s[i..after]);
                        i = after;
                        continue;
                    }
                    ConstructAction::NotConstruct => {}
                }
            }
        }

        // ── Separators ────────────────────────────────────────────
        let bytes = s.as_bytes();
        match c {
            '&' => {
                match bytes.get(i + 1) {
                    Some(b'&') => {
                        // `&&` — chain continues with threaded state.
                        validate_unit(
                            &current,
                            state,
                            pipeline_state.as_ref(),
                            in_pipeline,
                            negated || time_external_pending,
                        )?;
                        current.clear();
                        time_external_pending = false;
                        in_pipeline = false;
                        pipeline_state = None;
                        negated = false;
                        i += 2;
                    }
                    Some(b'>') => {
                        // `&>` / `&>>` redirect — not a separator.
                        current.push(c);
                        i += 1;
                    }
                    _ if ends_with_unescaped_redirect_op(&current) => {
                        // `>&`, `<&`, `2>&1` — redirect, not a separator.
                        current.push(c);
                        i += 1;
                    }
                    _ => {
                        // Single `&`: the whole chain ran in a background
                        // child — its state changes are discarded.
                        validate_unit(
                            &current,
                            state,
                            pipeline_state.as_ref(),
                            in_pipeline,
                            negated || time_external_pending,
                        )?;
                        current.clear();
                        time_external_pending = false;
                        in_pipeline = false;
                        pipeline_state = None;
                        negated = false;
                        *state = chain_start.snapshot();
                        chain_start = state.snapshot();
                        i += 1;
                    }
                }
            }
            '|' => {
                if bytes.get(i + 1) == Some(&b'|') {
                    // `||` — OR-list: the left side runs in the current shell
                    // and the right side only when the left fails. Thread
                    // state like `&&` (a failed left side left the real state
                    // unchanged, which matches the fail-closed tracking rules
                    // for failed commands). `||` is NOT a pipeline. Accepted
                    // residual: the right side is validated against the
                    // threaded state but runs exactly when the left FAILED
                    // (`cd /tmp || touch f` — touch lands in the real CWD);
                    // pre-existing baseline behavior, pinned by the `||`
                    // battery, not a hole this ticket closes.
                    validate_unit(
                        &current,
                        state,
                        pipeline_state.as_ref(),
                        in_pipeline,
                        negated || time_external_pending,
                    )?;
                    current.clear();
                    time_external_pending = false;
                    in_pipeline = false;
                    pipeline_state = None;
                    negated = false;
                    i += 2;
                } else if ends_with_unescaped_redirect_op(&current) {
                    // `>|` compound redirect — not a pipe.
                    current.push(c);
                    i += 1;
                } else {
                    // `|` / `|&` pipeline: members run in subshells. The
                    // FIRST member must flush against the pipeline-start
                    // snapshot too — set it before the flush. `negated`
                    // stays set: the whole pipeline is negated by a leading
                    // `!` (`! time cd /tmp | cat` — time is external in
                    // every member).
                    if pipeline_state.is_none() {
                        pipeline_state = Some(state.snapshot());
                    }
                    in_pipeline = true;
                    validate_unit(
                        &current,
                        state,
                        pipeline_state.as_ref(),
                        in_pipeline,
                        negated || time_external_pending,
                    )?;
                    current.clear();
                    time_external_pending = false;
                    i += if bytes.get(i + 1) == Some(&b'&') {
                        2
                    } else {
                        1
                    };
                }
            }
            ';' | '\n' => {
                validate_unit(
                    &current,
                    state,
                    pipeline_state.as_ref(),
                    in_pipeline,
                    negated || time_external_pending,
                )?;
                current.clear();
                time_external_pending = false;
                in_pipeline = false;
                pipeline_state = None;
                negated = false;
                chain_start = state.snapshot();
                i += c.len_utf8();
            }
            _ => {
                current.push(c);
                i += c.len_utf8();
            }
        }
    }
    validate_unit(
        &current,
        state,
        pipeline_state.as_ref(),
        in_pipeline,
        negated || time_external_pending,
    )?;
    Ok(s.len())
}

/// Validate the accumulated unit (a simple command or already-validated
/// construct span) against the effective state: pipeline members use a fresh
/// snapshot of the pipeline-start state (discarded — no leak); chain members
/// use the threaded `state` directly. `negated` marks a `!`-prefixed unit.
fn validate_unit(
    current: &str,
    state: &mut ValidationState,
    pipeline_state: Option<&ValidationState>,
    in_pipeline: bool,
    negated: bool,
) -> Result<(), String> {
    if in_pipeline {
        let mut snap = pipeline_state
            .expect("pipeline state set while in pipeline")
            .snapshot();
        validate_simple_command(current, &mut snap, negated)
    } else {
        validate_simple_command(current, state, negated)
    }
}

/// What to do with a token at command position.
enum ConstructAction {
    /// A compound construct was consumed and validated; continue after the
    /// returned index without accumulating text.
    Consumed(usize),
    /// A span that stays whole inside the current command (bare `((...))`
    /// arithmetic); accumulate `s[i..after]`.
    Accumulate(usize),
    /// Not a construct — accumulate normally.
    NotConstruct,
}

/// Advance `i` past leading whitespace (Unicode, including newlines — a
/// construct keyword may follow a newline: `if true\n then`). Returns `i`
/// unchanged at end of input.
fn skip_ws(s: &str, i: usize) -> usize {
    let mut k = i;
    while let Some(c) = s[k..].chars().next() {
        if !c.is_whitespace() {
            break;
        }
        k += c.len_utf8();
    }
    k
}

/// Read the next shell word at `i` (skipping leading whitespace, including
/// newlines — construct keywords can follow a newline: `if true\n then`).
/// Returns `None` when the word is quoted (a quoted keyword is a literal
/// command name, not a keyword) or when there is no word.
fn read_keyword_at(s: &str, i: usize) -> Option<String> {
    let mut k = skip_ws(s, i);
    let mut word = String::new();
    while k < s.len() {
        let c = s[k..].chars().next().expect("k < len");
        if c.is_whitespace() || matches!(c, ';' | '&' | '|' | '\n') {
            break;
        }
        if matches!(c, '\'' | '"' | '$' | '`' | '\\') {
            return None; // quoted / substitution-formed — not a keyword
        }
        word.push(c);
        k += c.len_utf8();
    }
    (!word.is_empty()).then_some(word)
}

/// Dispatch a token at command position: compound constructs, stray
/// terminators (reject), `{`/`(` groups, arithmetic. (`!` negation is handled
/// by [`scan_list`] itself — it must set the negated-context flag.)
///
/// `base` is the state the construct's parts snapshot from. For chain members
/// it is the threaded state; for pipeline members the caller passes a fresh
/// snapshot (which is discarded after the construct — pipeline members run in
/// subshells). Brace groups thread `base` (they run in the current shell).
fn handle_construct(
    s: &str,
    i: usize,
    base: &mut ValidationState,
) -> Result<ConstructAction, String> {
    let Some(word) = read_keyword_at(s, i) else {
        return Ok(ConstructAction::NotConstruct);
    };
    match word.as_str() {
        "if" => Ok(ConstructAction::Consumed(parse_if(s, i, base)?)),
        "for" | "select" => Ok(ConstructAction::Consumed(parse_for(s, i, base)?)),
        "while" | "until" => Ok(ConstructAction::Consumed(parse_while_until(s, i, base)?)),
        "case" => Ok(ConstructAction::Consumed(parse_case(s, i, base)?)),
        "function" => Ok(ConstructAction::Consumed(parse_function(s, i, base)?)),
        // Stray terminators at command position — the construct they close is
        // not open here (fail-closed; bash treats them as syntax errors).
        "fi" | "done" | "esac" | "then" | "do" | "elif" | "else" | "}" | ")" => reject(
            &s[i..],
            "a shell control keyword appears without its matching construct.",
            "remove the stray keyword, or complete the construct it belongs to.",
        ),
        _ => {
            let c = s[i..].chars().next().expect("i < len");
            if c == '{' && is_brace_group_start(s, i) {
                Ok(ConstructAction::Consumed(parse_brace(s, i, base)?))
            } else if c == '(' {
                // `((` is the arithmetic command (kept whole, validated as a
                // simple command); `(` is a subshell.
                if s.as_bytes().get(i + 1) == Some(&b'(') {
                    Ok(ConstructAction::Accumulate(find_arithmetic_end(s, i)))
                } else {
                    Ok(ConstructAction::Consumed(parse_subshell(s, i, base)?))
                }
            } else {
                // `name()` / `name ()` function definition (POSIX form).
                if looks_like_function_def(s, i) {
                    Ok(ConstructAction::Consumed(parse_function(s, i, base)?))
                } else {
                    Ok(ConstructAction::NotConstruct)
                }
            }
        }
    }
}

/// True when the word at `i` is an identifier immediately followed by `()`
/// (optionally with whitespace between): a POSIX function definition.
fn looks_like_function_def(s: &str, i: usize) -> bool {
    let mut k = i;
    let mut name_len = 0usize;
    while k < s.len() {
        let c = s[k..].chars().next().expect("k < len");
        if !(c.is_ascii_alphanumeric() || c == '_') {
            break;
        }
        k += c.len_utf8();
        name_len += 1;
    }
    if name_len == 0 {
        return false;
    }
    k = skip_ws(s, k);
    s[k..].starts_with("()")
}

/// Read one shell word at `i` (quote/substitution aware, skipping leading
/// whitespace). Returns `(word_text, index_after)`.
fn read_word_at(s: &str, i: usize) -> (String, usize) {
    let mut k = skip_ws(s, i);
    let mut word = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    while k < s.len() {
        let c = s[k..].chars().next().expect("k < len");
        if c == '\\' && !in_single {
            let Some(next) = s[k + c.len_utf8()..].chars().next() else {
                word.push(c);
                k += c.len_utf8();
                break;
            };
            word.push(c);
            word.push(next);
            k += c.len_utf8() + next.len_utf8();
            continue;
        }
        if !escaped
            && !in_single
            && let Some((_, next)) = substitution_span(s, k)
        {
            word.push_str(&s[k..next]);
            k = next;
            continue;
        }
        if !super::track_char_context(c, &mut in_single, &mut in_double, &mut escaped) {
            word.push(c);
            k += c.len_utf8();
            continue;
        }
        if c.is_whitespace() || matches!(c, ';' | '&' | '|' | '\n') {
            break;
        }
        word.push(c);
        k += c.len_utf8();
    }
    (word, k)
}

/// Find the `do` keyword after a for/while header. Skips `;`/newline
/// separators and whitespace; returns the index after `do`, or `Err` when
/// the header is unterminated.
fn find_do_keyword(s: &str, mut i: usize) -> Result<usize, String> {
    loop {
        i = skip_ws(s, i);
        if i >= s.len() {
            return reject(
                s,
                "a for/while/until/select construct is missing its `do` keyword.",
                "write the construct with its closing `done`.",
            );
        }
        let c = s[i..].chars().next().expect("i < len");
        if matches!(c, ';' | '\n') {
            i += c.len_utf8();
            continue;
        }
        break;
    }
    match read_keyword_at(s, i) {
        Some(w) if w == "do" => Ok(i + 2),
        _ => reject(
            s,
            "a for/while/until/select construct is missing its `do` keyword.",
            "write the construct with its closing `done`.",
        ),
    }
}

/// The stop keyword/token that ended a [`scan_list`] call returning `after`
/// (the index right after the keyword). Used by the construct parsers to
/// dispatch on which terminator stopped the scan.
fn stop_keyword_at(s: &str, after: usize) -> Option<&str> {
    for kw in ["elif", "else", "esac", "done", "then", "fi", "do", "}", ")"] {
        if after >= kw.len() && &s[after - kw.len()..after] == kw {
            return Some(kw);
        }
    }
    None
}

// ── Compound construct parsers ─────────────────────────────────────────

/// A cd executed inside a construct part runs in the current shell (if/for/
/// while/case/select bodies and conditions leak to the parent) — the leaked
/// CWD is untrackable, so the outer tracking resets fail-closed. The counter
/// propagates so enclosing constructs detect the cd too. Conservative by
/// design: even a temp-target cd in a body resets (the guard cannot prove the
/// construct executes at runtime), which over-rejects safe spellings but never
/// under-rejects.
///
/// Variable bindings made inside the part leak at runtime the same way
/// (`if true; then export TMPDIR=/etc; fi; touch $TMPDIR/f` writes to /etc).
/// They are merged conservatively: a binding that differs from the outer one
/// is uncertain (the construct may not take that branch), so the variable
/// poisons — its expansions reject fail-closed. A variable FIRST bound inside
/// a part is uncertain too (the branch may not execute, leaving it unset at
/// runtime — `if false; then D=/tmp; fi; touch $D/f` writes to `/f`), so it
/// also poisons, mirroring the cd reset.
fn note_part_cd(part: &ValidationState, base: &mut ValidationState) {
    if part.cd_count != base.cd_count {
        base.cwd = None;
        base.cd_count = part.cd_count;
    }
    for (name, pb) in &part.vars {
        match base.vars.get(name) {
            Some(bb) if !bb.poisoned && !pb.poisoned && bb.value == pb.value => {}
            // Unknown (first bound inside the part — the branch may not
            // execute) or diverging binding: uncertain, poison fail-closed.
            None | Some(_) => {
                base.vars.insert(
                    name.clone(),
                    VarBinding {
                        poisoned: true,
                        value: pb.value.clone(),
                    },
                );
            }
        }
    }
}

/// Parse an `if`/`elif`/`else` construct starting at `i` (at `if`). The whole
/// construct is a conditional boundary: every part is validated against a
/// snapshot of the pre-construct state and nothing leaks past `fi`.
fn parse_if(s: &str, i: usize, base: &mut ValidationState) -> Result<usize, String> {
    let after_if = i + 2;
    let mut cond = base.snapshot();
    // `time` is external at the first word of the condition (probe-verified:
    // `if time cd ...` runs /usr/bin/time; the timed cd is in a child).
    let then_idx = scan_list(s, after_if, &["then"], &[], &mut cond, true)?;
    note_part_cd(&cond, base);
    let mut body = base.snapshot();
    let mut after_part = scan_list(s, then_idx, &["elif", "else", "fi"], &[], &mut body, false)?;
    note_part_cd(&body, base);
    loop {
        match stop_keyword_at(s, after_part) {
            Some("elif") => {
                let mut econd = base.snapshot();
                let ethen = scan_list(s, after_part, &["then"], &[], &mut econd, true)?;
                note_part_cd(&econd, base);
                let mut ebody = base.snapshot();
                after_part = scan_list(s, ethen, &["elif", "else", "fi"], &[], &mut ebody, false)?;
                note_part_cd(&ebody, base);
            }
            Some("else") => {
                let mut eb = base.snapshot();
                let fi = scan_list(s, after_part, &["fi"], &[], &mut eb, false)?;
                note_part_cd(&eb, base);
                return Ok(fi);
            }
            Some("fi") => return Ok(after_part),
            _ => {
                return reject(
                    s,
                    "an if construct is missing its closing `fi`.",
                    "write the construct with its closing `fi`.",
                );
            }
        }
    }
}

/// Parse a `while`/`until` construct: condition until `do`, body until
/// `done`. The whole construct is a conditional boundary — nothing leaks.
fn parse_while_until(s: &str, i: usize, base: &mut ValidationState) -> Result<usize, String> {
    let after_kw = i + 5;
    let mut cond = base.snapshot();
    // `time` is external at the first word of the condition (`while time cd
    // ...` runs /usr/bin/time; the timed cd is in a child).
    let do_idx = scan_list(s, after_kw, &["do"], &[], &mut cond, true)?;
    note_part_cd(&cond, base);
    if stop_keyword_at(s, do_idx) != Some("do") {
        return reject(
            s,
            "a while/until construct is missing its `do` keyword.",
            "write the construct with its closing `done`.",
        );
    }
    let mut body = base.snapshot();
    let done_idx = scan_list(s, do_idx, &["done"], &[], &mut body, false)?;
    note_part_cd(&body, base);
    if stop_keyword_at(s, done_idx) != Some("done") {
        return reject(
            s,
            "a while/until construct is missing its closing `done`.",
            "write the construct with its closing `done`.",
        );
    }
    Ok(done_idx)
}

/// Parse a `for`/`select` construct. `for name [in words]; do body; done` or
/// `for ((arith)); do body; done`. The loop variable is temp-bound only when
/// every static prefix in the header word list provably resolves under a temp
/// root; the body is validated from a pre-construct snapshot plus that
/// binding; nothing leaks past `done`.
fn parse_for(s: &str, i: usize, base: &mut ValidationState) -> Result<usize, String> {
    let after_kw = i + if s[i..].starts_with("select") { 6 } else { 3 };
    let (name, after_name) = read_word_at(s, after_kw);
    if name.starts_with("((") {
        // Arithmetic header: `for (( init; cond; incr )); do`. The header is
        // an expression, not commands — but command substitutions inside it
        // execute and must be validated.
        let first_paren = after_name - name.len();
        let after_arith = find_arithmetic_end(s, first_paren);
        let mut header_state = base.snapshot();
        scan_substitutions(&s[first_paren..after_arith], &mut header_state)?;
        let do_idx = find_do_keyword(s, after_arith)?;
        let mut body = base.snapshot();
        let done_idx = scan_list(s, do_idx, &["done"], &[], &mut body, false)?;
        note_part_cd(&body, base);
        if stop_keyword_at(s, done_idx) != Some("done") {
            return reject(
                s,
                "a for/select construct is missing its closing `done`.",
                "write the construct with its closing `done`.",
            );
        }
        return Ok(done_idx);
    }
    if name.is_empty() {
        return reject(
            s,
            "a for/select construct is missing its loop variable.",
            "write `for name in ...; do ...; done`.",
        );
    }
    let mut words: Vec<String> = Vec::new();
    let mut pos = after_name;
    // Optional `in` keyword — read it with its true extent (read_word_at
    // skips leading whitespace; advancing by the keyword length from the
    // pre-whitespace position would land mid-keyword).
    let (first_word, after_first) = read_word_at(s, pos);
    if first_word == "in" {
        pos = after_first;
        // Collect the word list until `;`/newline (the `do` follows).
        loop {
            pos = skip_ws(s, pos);
            if pos >= s.len() {
                break;
            }
            let ch = s[pos..].chars().next().expect("pos < len");
            if matches!(ch, ';' | '\n') {
                break;
            }
            if ch == '(' && s.as_bytes().get(pos + 1) == Some(&b'(') {
                // A bare `((` inside the word list is malformed — reject.
                return reject(
                    s,
                    "a for/select header contains an unexpected arithmetic span.",
                    "write the header as `for name in words; do ...; done`.",
                );
            }
            let (word, after_word) = read_word_at(s, pos);
            if word.is_empty() {
                break;
            }
            // `do` at the start of a word is a LIST WORD on the runtime shell
            // (bash 3.2: `for x in do; do ...` iterates over `do`) — only
            // `;`/newline end the list. POSIX requires that separator before
            // `do`, and bash rejects `for x in a b do ...` as a syntax error,
            // so consuming `do` here and failing `find_do_keyword` below is
            // the correct fail-closed outcome for that spelling.
            words.push(word);
            pos = after_word;
        }
        // Command substitutions in the word list execute at loop start and
        // must be validated (`for x in $(touch f); do ...`).
        let mut header_state = base.snapshot();
        for word in &words {
            scan_substitutions(word, &mut header_state)?;
        }
    }
    let do_idx = find_do_keyword(s, pos)?;
    let mut body = base.snapshot();
    if let Some(binding) = loop_var_binding(&words, base) {
        body.vars.insert(name, binding);
    }
    let done_idx = scan_list(s, do_idx, &["done"], &[], &mut body, false)?;
    note_part_cd(&body, base);
    if stop_keyword_at(s, done_idx) != Some("done") {
        return reject(
            s,
            "a for/select construct is missing its closing `done`.",
            "write the construct with its closing `done`.",
        );
    }
    Ok(done_idx)
}

/// Temp-binding for a for-loop variable: only when every static prefix of the
/// header word list provably resolves under a temp root (`for f in /tmp/*.db`)
/// is the variable temp-bound; a non-temp prefix (`for db in ~/.mahbot/db/*.db`)
/// leaves it unbound (existing fail-closed semantics apply).
fn loop_var_binding(words: &[String], state: &ValidationState) -> Option<VarBinding> {
    if words.is_empty() {
        return None;
    }
    words
        .iter()
        .all(|w| {
            // Static prefix: up to the first glob metacharacter.
            let prefix = w.find(['*', '?', '[']).map_or(w.as_str(), |g| &w[..g]);
            if prefix.is_empty() {
                return false;
            }
            resolve_path_word(prefix, state).is_some_and(|p| {
                is_path_under_temp(&crate::tools::path::normalize_path(&p), state.ctx)
            })
        })
        .then_some(VarBinding {
            value: None,
            poisoned: false,
        })
}

/// Parse a `case word in pattern) body ;; ... esac` construct. The subject
/// and patterns are validated for command substitutions (bash executes them);
/// each body is validated from a pre-construct snapshot; nothing leaks past
/// `esac`. `;;`, `;&`, and `;;&` terminate bodies.
fn parse_case(s: &str, i: usize, base: &mut ValidationState) -> Result<usize, String> {
    let after_case = i + 4;
    let (subject, mut k) = read_word_at(s, after_case);
    if subject.is_empty() {
        return reject(
            s,
            "a case construct is missing its subject word.",
            "write `case word in pattern) body ;; esac`.",
        );
    }
    let mut subj_state = base.snapshot();
    scan_substitutions(&subject, &mut subj_state)?;
    // Find the `in` keyword at command position.
    loop {
        k = skip_ws(s, k);
        if k >= s.len() {
            return reject(
                s,
                "a case construct is missing its `in` keyword.",
                "write `case word in ... esac`.",
            );
        }
        let c = s[k..].chars().next().expect("k < len");
        if matches!(c, ';' | '\n') {
            k += c.len_utf8();
            continue;
        }
        break;
    }
    let in_word = read_keyword_at(s, k);
    if in_word.as_deref() != Some("in") {
        return reject(
            s,
            "a case construct is missing its `in` keyword.",
            "write `case word in ... esac`.",
        );
    }
    k += 2;
    // Pattern/body loop.
    loop {
        // Pattern section: until an unquoted `)`.
        let (pat, after_pat) = scan_case_pattern(s, k)?;
        let mut pat_state = base.snapshot();
        scan_substitutions(&pat, &mut pat_state)?;
        // Body until `;;`/`;&`/`;;&` (or `esac` for an empty last body).
        let mut body = base.snapshot();
        // `time` is external at the first word of the arm body (probe-
        // verified: `case x in x) time cd ...` runs /usr/bin/time).
        let after_body = scan_list(
            s,
            after_pat,
            &["esac"],
            &[";;", ";&", ";;&"],
            &mut body,
            true,
        )?;
        note_part_cd(&body, base);
        // The body scan stopped at `esac` (empty last body) or a `;;`-family
        // terminator — either way, `esac` must close the construct next.
        if stop_keyword_at(s, after_body) == Some("esac") {
            return Ok(after_body);
        }
        let j = skip_ws(s, after_body);
        if read_keyword_at(s, j).as_deref() == Some("esac") {
            return Ok(j + 4);
        }
        // Next pattern.
        k = after_body;
    }
}

/// Scan a case pattern section from `i` until an unquoted `)` that ends the
/// pattern list. Patterns may span newlines and contain `|` alternation (not
/// a pipeline separator here). Returns `(pattern_text, index_after_paren)`.
fn scan_case_pattern(s: &str, i: usize) -> Result<(String, usize), String> {
    let mut k = i;
    let mut pat = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    while k < s.len() {
        let c = s[k..].chars().next().expect("k < len");
        if c == '\\' && !in_single {
            let Some(next) = s[k + c.len_utf8()..].chars().next() else {
                return reject(
                    s,
                    "a case pattern is unterminated.",
                    "write `case word in pattern) body ;; esac`.",
                );
            };
            pat.push(c);
            pat.push(next);
            k += c.len_utf8() + next.len_utf8();
            continue;
        }
        if !escaped
            && !in_single
            && let Some((_, next)) = substitution_span(s, k)
        {
            pat.push_str(&s[k..next]);
            k = next;
            continue;
        }
        if !super::track_char_context(c, &mut in_single, &mut in_double, &mut escaped) {
            pat.push(c);
            k += c.len_utf8();
            continue;
        }
        if c == ')' {
            return Ok((pat, k + 1));
        }
        pat.push(c);
        k += c.len_utf8();
    }
    reject(
        s,
        "a case pattern is unterminated.",
        "write `case word in pattern) body ;; esac`.",
    )
}

/// `{` opens a brace group only as a reserved word — bash requires it to be
/// delimited by whitespace or a metacharacter on the right (`{touch,}` is a
/// single brace-expansion word, not a group; `}` does not delimit — `{}` is
/// a syntax error). An undelimited `{` falls through to the segment validator,
/// where a `{`-containing verb is unprovable (fail-closed).
fn is_brace_group_start(s: &str, i: usize) -> bool {
    match s[i + 1..].chars().next() {
        None => true, // `{` at end of input — unterminated, rejected below
        Some(c) => c.is_whitespace() || matches!(c, ';' | '&' | '|' | '(' | ')' | '<' | '>'),
    }
}

/// Parse a `{ body; }` brace group starting at `i`. Brace groups run in the
/// current shell: the pre-group state threads through the body (validated as
/// commands — mutators reject, temp-scoped writes stay allowed). Any
/// `cd`/`pushd`/`popd` inside resets the tracked CWD after the group
/// (conservative: the cd moved the real CWD behind the guard).
fn parse_brace(s: &str, i: usize, state: &mut ValidationState) -> Result<usize, String> {
    let body_start = i + 1;
    let entry_count = state.cd_count;
    let after_body = scan_list(s, body_start, &[], &["}"], state, false)?;
    if stop_keyword_at(s, after_body) != Some("}") {
        return reject(
            s,
            "a brace group is missing its closing `}`.",
            "write `{ ...; }` with the closing brace.",
        );
    }
    // A `cd`/`pushd`/`popd` executed in the group — including one hidden in a
    // quoted eval body (`{ eval 'cd /tmp'; }`) — moves the real CWD behind the
    // guard: reset tracking fail-closed (conservative, pinned). Position-aware:
    // only command-position cd verbs bump the counter (`{ echo cd; }` does not).
    if state.cd_count != entry_count {
        state.cwd = None;
    }
    Ok(after_body)
}

/// Parse a `( body )` subshell starting at `i`. The body is validated from a
/// pre-construct snapshot and nothing leaks past the closing paren.
fn parse_subshell(s: &str, i: usize, base: &ValidationState) -> Result<usize, String> {
    let mut body = base.snapshot();
    let after_body = scan_list(s, i + 1, &[], &[")"], &mut body, false)?;
    if stop_keyword_at(s, after_body) != Some(")") {
        return reject(
            s,
            "a subshell is missing its closing `)`.",
            "write `( ... )` with the closing parenthesis.",
        );
    }
    Ok(after_body)
}

/// Parse a function definition starting at `i`: `function name { ... }`,
/// `name() { ... }`, or `name () ( ... )`. The body is a compound command
/// validated from a pre-construct snapshot; a function body is non-leaking
/// (it does not execute at definition time).
fn parse_function(s: &str, i: usize, base: &ValidationState) -> Result<usize, String> {
    let base = base.snapshot();
    let mut k = i;
    if s[k..].starts_with("function") {
        k += 8;
    }
    // Read the function name.
    let (name, after_name) = read_word_at(s, k);
    if name.is_empty() {
        return reject(
            s,
            "a function definition is missing its name.",
            "write `name() { ...; }`.",
        );
    }
    k = after_name;
    // Optional `()` between the name and the body.
    k = skip_ws(s, k);
    if s[k..].starts_with("()") {
        k += 2;
    }
    // The body opens with `{` (or `(` — a subshell-bodied function).
    k = skip_ws(s, k);
    let c = s[k..].chars().next().expect("k < len");
    let after_body = if c == '{' {
        let mut body = base.snapshot();
        scan_list(s, k + 1, &[], &["}"], &mut body, false)?
    } else if c == '(' {
        let mut body = base.snapshot();
        scan_list(s, k + 1, &[], &[")"], &mut body, false)?
    } else {
        return reject(
            s,
            "a function definition is missing its body.",
            "write `name() { ...; }`.",
        );
    };
    Ok(after_body)
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
            && let Some((content, next)) = substitution_span(s, i)
        {
            // `$((...))` arithmetic expansion: the expression is not a
            // command, but nested substitutions inside it execute and
            // must be validated (e.g. `$(( $(touch x) + 1 ))`). The `$(` gate
            // excludes backtick bodies starting with `(` (a subshell command).
            if c == '$' && content.starts_with('(') {
                scan_substitutions(content, state)?;
            } else {
                validate_substitution_content(content, state)?;
            }
            i = next;
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

/// Scan from byte `from` for the close paren that brings `initial_depth` to 0.
/// Quote- and escape-aware (via track_char_context); every unquoted paren
/// nests — `$((` arithmetic spans, subshell parens, and nested `$(` alike —
/// so `$(( (a) * (b) ))` and `$( (echo hi) )` close correctly. Returns the
/// byte index AFTER the closing paren, or None when unterminated (fail-closed).
fn find_paren_close(s: &str, from: usize, initial_depth: usize) -> Option<usize> {
    let mut depth = initial_depth;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut i = from;
    while i < s.len() {
        let c = s[i..].chars().next().expect("i < s.len()");
        if !super::track_char_context(c, &mut in_single, &mut in_double, &mut escaped) {
            i += c.len_utf8();
            continue;
        }
        if c == '(' {
            depth += 1;
        } else if c == ')' {
            depth -= 1;
            if depth == 0 {
                return Some(i + 1);
            }
        }
        i += c.len_utf8();
    }
    None
}

/// Find the matching close paren for a `$(` substitution whose content starts
/// at byte `start`. Returns `(content, index_after_close)`; unterminated
/// substitutions return the rest as content (still validated — fail-closed).
fn find_substitution_end(s: &str, start: usize) -> (&str, usize) {
    match find_paren_close(s, start, 1) {
        Some(end) => (&s[start..end - 1], end),
        None => (&s[start..], s.len()),
    }
}

/// Skip a bare `((...))` arithmetic span whose first `(` is at byte `i`.
/// Returns the index after the closing `))`, or `s.len()` when unterminated.
fn find_arithmetic_end(s: &str, i: usize) -> usize {
    find_paren_close(s, i + 2, 2).unwrap_or(s.len())
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

/// Span of a command substitution whose introducer is at byte `i` (`$(` or
/// backtick). Returns `(content, index_after_close)` — `content` starts
/// AFTER the introducer (`i + 2` / `i + 1`), not at `i`; slice `s[i..next]`
/// for the full span.
fn substitution_span(s: &str, i: usize) -> Option<(&str, usize)> {
    let b = s.as_bytes();
    if b.get(i) == Some(&b'$') && b.get(i + 1) == Some(&b'(') {
        Some(find_substitution_end(s, i + 2))
    } else if b.get(i) == Some(&b'`') {
        Some(find_backtick_end(s, i + 1))
    } else {
        None
    }
}

/// Validate a substitution body as a nested command against a snapshot of the
/// current state (substitutions run in a subshell — state changes don't leak
/// to the outer command).
fn validate_substitution_content(inner: &str, state: &mut ValidationState) -> Result<(), String> {
    let mut snapshot = state.snapshot();
    validate_string(inner, &mut snapshot)
}

/// Construct a read-only rejection error with consistent formatting. The Ok
/// type is generic so any `Result<T, String>` validator can return it.
fn reject<T>(cmd: &str, why: &str, suggestion: &str) -> Result<T, String> {
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

/// One temp-gated mutator dispatch row.
struct MutatorCheck {
    verbs: &'static [&'static str],
    /// Reject predicate: true when the invocation writes outside temp and must
    /// be rejected; `None` = always rejected (unconditional mutator).
    rejects: Option<fn(&str, &str, &ValidationState) -> bool>,
    /// Rejection reason template with a `{verb}` placeholder.
    rejection: &'static str,
    /// Suggestion strings: the first educates about recognized temp-variable
    /// spellings when a `$` path failed to resolve; the second is the generic
    /// read-only-alternatives fallback.
    suggestions: (&'static str, &'static str),
}

/// True when a scratch mutator writes outside temp.
fn scratch_rejects(segment: &str, _verb: &str, state: &ValidationState) -> bool {
    !scratch_paths_under_temp(segment, state)
}

/// True when a temp mutator writes outside temp; `cp` only needs its
/// destination under temp (sources are read-only — copying from anywhere
/// into temp is allowed; copying into the workspace stays blocked).
fn temp_rejects(segment: &str, verb: &str, state: &ValidationState) -> bool {
    !temp_mutator_paths_under_temp(segment, verb, state)
}

const READONLY_ALTERNATIVES: &str = "use read-only alternatives to inspect files, e.g. `cat`, `head`, `tail`, `ls`, `file`, `stat`.";

/// Temp-gated mutator dispatch, iterated in order by [`check_segment`]:
/// scratch mutators (tee/touch/mkdir), temp mutators (cp/mv/rm/…), then the
/// unconditional mutators (always rejected).
const MUTATOR_CHECKS: &[MutatorCheck] = &[
    MutatorCheck {
        verbs: SCRATCH_MUTATORS,
        rejects: Some(scratch_rejects),
        rejection: "`{verb}` is not allowed outside temp directories — it modifies the workspace.",
        suggestions: (
            "use a literal path under /tmp, or bind the directory first with `NAME=$(mktemp -d)` and reference `$NAME`.",
            READONLY_ALTERNATIVES,
        ),
    },
    MutatorCheck {
        verbs: TEMP_MUTATORS,
        rejects: Some(temp_rejects),
        rejection: "`{verb}` is not allowed outside temp directories — it modifies files outside /tmp.",
        suggestions: (
            "use a literal path under /tmp, or bind the directory first with `NAME=$(mktemp -d)` and reference `$NAME`.",
            "use paths under /tmp, /var/tmp, or the OS temp directory, or use read-only alternatives like `cat`, `head`, `tail`, `ls`, `file`, `stat`.",
        ),
    },
    MutatorCheck {
        verbs: MUTATING_COMMANDS,
        rejects: None,
        rejection: "`{verb}` is not allowed — it modifies the workspace.",
        suggestions: (READONLY_ALTERNATIVES, READONLY_ALTERNATIVES),
    },
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
        rejection: "`sed` in-place editing (`-i`/`-I`/`--in-place`) is not allowed outside temp directories — it modifies files in-place.",
        suggestion: "use `sed` without in-place flags to output to stdout, e.g. `sed 's/a/b/' file`, or use `-i` with a path under /tmp.",
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
        rejection: "`base64` with `-o`/`--output` is not allowed outside temp directories — it writes output to a file.",
        suggestion: "use `base64` without `-o` to output to stdout, or use `-o /tmp/...` to write to temp.",
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
pub(super) enum TokenKind {
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
/// This is a **token-level** classifier shared by [`non_flag_path_args`] and
/// the grep engine's rewrite path (which re-creates redirect tokens verbatim).
/// It is NOT used by [`has_disallowed_redirect`], which operates at a different
/// abstraction level (character-based with quote-state awareness).  Those two
/// functions have distinct semantics and are deliberately kept separate.
pub(super) fn classify_shell_token(w: &str) -> TokenKind {
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
    !paths.is_empty() && paths.iter().all(|p| !writes_outside_temp(p, state))
}

/// True when a temp mutator's path arguments all resolve under an allowed
/// temp root. `cp` is special-cased: only the destination must be under temp
/// (sources are read-only — copying from anywhere into temp is allowed;
/// copying into the workspace stays blocked).
fn temp_mutator_paths_under_temp(segment: &str, first_word: &str, state: &ValidationState) -> bool {
    if first_word == "cp" {
        return cp_destination_under_temp(segment, state);
    }
    scratch_paths_under_temp(segment, state)
}

/// Identify the `cp` destination: the value of
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
            return !writes_outside_temp(val, state);
        }
    }
    if let Some(val) = flag_value_equals(rest, "--target-directory=") {
        return !writes_outside_temp(val, state);
    }

    match non_flag_path_args(segment).last() {
        Some(dest) => !writes_outside_temp(dest, state),
        None => false, // no identifiable destination → reject
    }
}

/// True when the trimmed segment is exactly one command-substitution span
/// (`$(...)` or backtick) with no leading/trailing content — a bare
/// substitution at command position, whose content executes in a subshell and
/// is validated by [`scan_substitutions`] (the emitted heredoc-body spans and
/// standalone `$(cmd)` commands).
fn is_bare_substitution_segment(s: &str) -> bool {
    let s = s.trim();
    substitution_span(s, 0).is_some_and(|(_, next)| next == s.len())
}

/// Classification of a command verb word: provably literal (possibly with
/// balanced surrounding quotes) or unprovable (concatenated quotes, ANSI-C
/// quoting, escapes, substitutions — the guard cannot prove what the word
/// resolves to without full shell word-expansion modeling).
#[derive(Debug, Clone, PartialEq, Eq)]
enum VerbClass<'a> {
    Literal(&'a str),
    Unprovable,
}

/// Result of scanning a segment's leading env assignments and shell prefixes.
#[derive(Debug, Clone, PartialEq, Eq)]
enum VerbResolution<'a> {
    /// Verb word found at `idx` in the whitespace-split words.
    Verb { idx: usize, class: VerbClass<'a> },
    /// A forwarding prefix with informational/invalid options (`command -v`,
    /// `builtin -p`, `time -p -p`): the command is provably never executed,
    /// so the segment leaves the real CWD and environment untouched.
    Informational,
    /// No verb word (empty segment, only env assignments).
    None,
}

/// First effective command word of a segment: after env assignments and the
/// forwarding prefixes (`command`/`builtin`, plus `time` at the head), each
/// prefix's option grammar decides whether the command executes:
///
/// - `command`: `-p` (repeatable, including combined `-pp`) forwards; `-v`/`-V`
///   (alone or combined) and invalid options are informational (the command is
///   described or the option errors — it never executes); `--` ends options
///   and forwards.
/// - `builtin`: `--` forwards; any other option is invalid → informational.
/// - `time`: forwards ONLY as the unquoted first word of the full segment,
///   outside a `!`-negated list and outside a condition/case-arm head (the
///   first word of an `if`/`elif`/`while`/`until` condition or case-arm
///   body — `if time cd /tmp` runs external time with the timed cd in a
///   child). Quoted `"time"`, an env assignment or `command`/`builtin`
///   before it, a nested `time time`, or `! time` all resolve to the
///   external `time` command — its timed command runs in a child, so the
///   guard must not track a cwd the shell does not reach. At most one
///   UNQUOTED `-p` follows (a quoted `"-p"` is the timed command, not an
///   option — keyword parsing); any further word (including `--`, a second
///   `-p`, or `-v`) IS the timed command, and a `-`-prefixed one does not
///   exist (`-p: command not found`) → informational.
///
/// Prefixes may be composed (`command builtin cd`, `time command cd`,
/// `builtin -- command cd`) — each level forwards to the current shell, so
/// the loop continues until a non-forwarding verb is found. Non-forwarding
/// prefixes (`env`/`sudo`/`nice`/…) run in a child shell and are returned as
/// the verb (their command's CWD change never reaches the parent).
///
/// Leading env assignments (`TMPDIR=/tmp cd /tmp`) are valid before the first
/// prefix, and after `time` (whose operand is a full command list — `time
/// FOO=bar cd /tmp` runs the cd). After `command`/`builtin` a `FOO=bar` word
/// is the COMMAND NAME (`command FOO=bar cd /tmp` errors 127 and the cd never
/// runs), so assignments are never skipped there.
///
/// The verb word is classified by [`classify_verb_word`]: balanced-quoted
/// words (`"cd"`) stay literal and modeled; concatenated (`"c"d`, `c'd'`),
/// ANSI-C (`$'cd'`), escaped (`c\144`), or substitution-formed verbs are
/// unprovable. `eval` is not skipped: its body is handled as a current-shell
/// construct.
fn resolve_verb<'a>(words: &[&'a str], negated: bool) -> VerbResolution<'a> {
    let mut i = 0;
    // Leading env assignments are valid before the first prefix
    // (`TMPDIR=/tmp cd /tmp` runs the cd). After `command`/`builtin` a
    // `FOO=bar` word is the COMMAND NAME (`command FOO=bar cd /tmp` errors
    // with 127 — the cd never runs), so assignments are skipped only here
    // and after `time`, whose operand is a full command list (`time FOO=bar
    // cd /tmp` runs the cd).
    while i < words.len() && super::is_env_assignment(words[i]) {
        i += 1;
    }
    loop {
        if i >= words.len() {
            return VerbResolution::None;
        }
        let w = words[i];
        let unquoted = strip_outer_quotes(w).map_or("", |(c, _)| c);
        if !matches!(unquoted, "command" | "builtin" | "time") {
            return VerbResolution::Verb {
                idx: i,
                class: classify_verb_word(w),
            };
        }
        let opts = &words[i + 1..];
        let mut j = 0;
        let executes = match unquoted {
            "command" => loop {
                let Some(o) = opts.get(j) else {
                    return VerbResolution::Informational;
                };
                let oq = strip_outer_quotes(o).map_or("", |(c, _)| c);
                if oq == "--" {
                    j += 1;
                    break true;
                }
                // `-p` is repeatable and combined short flags forward
                // (`-pp`, `-ppp` execute the command; `-pv`/`-pV` are
                // informational — `-v`/`-V` describe instead of run).
                if oq.starts_with('-') && oq.len() > 1 && oq[1..].bytes().all(|b| b == b'p') {
                    j += 1;
                    continue;
                }
                if oq.starts_with('-') && oq.len() > 1 {
                    return VerbResolution::Informational;
                }
                break true;
            },
            "builtin" => {
                let Some(o) = opts.first() else {
                    return VerbResolution::Informational;
                };
                let oq = strip_outer_quotes(o).map_or("", |(c, _)| c);
                if oq == "--" {
                    j = 1;
                    true
                } else if oq.starts_with('-') && oq.len() > 1 {
                    return VerbResolution::Informational;
                } else {
                    true
                }
            }
            _ => {
                // `time` is the reserved word only as the unquoted first word
                // of the FULL segment (`i == 0`, raw word `time`), outside a
                // `!`-negated list and outside a condition/case-arm head.
                // Quoted `"time"`, an env assignment or `command`/`builtin`
                // before it, a nested `time time`, and `! time` (or `time`
                // right after `if`/`elif`/`while`/`until`/`)`) all resolve to
                // the external `time` command — its timed command runs in a
                // child, so the real CWD never changes and tracking would
                // approve a chained write that lands in the workspace
                // (fail-closed: return it as a verb).
                if i != 0 || w != "time" || negated {
                    return VerbResolution::Verb {
                        idx: i,
                        class: classify_verb_word(w),
                    };
                }
                // `time` consumes at most one `-p`, matched on the RAW word:
                // keyword parsing treats a quoted `"-p"` as the timed command
                // (`-p: command not found`, nothing executes), not as the
                // option. Anything after the option is the timed command; a
                // `-`-prefixed one does not exist (`time -p -p` →
                // `-p: command not found`), so nothing executes.
                if opts.first().is_some_and(|o| *o == "-p") {
                    j += 1;
                }
                if opts.get(j).is_some_and(|o| {
                    let oq = strip_outer_quotes(o).map_or("", |(c, _)| c);
                    oq.starts_with('-') && oq.len() > 1
                }) {
                    return VerbResolution::Informational;
                }
                true
            }
        };
        if !executes {
            return VerbResolution::Informational;
        }
        let idx = i + 1 + j;
        if idx >= words.len() {
            return VerbResolution::None;
        }
        // The forwarded verb may itself be a forwarding prefix — continue the
        // loop so composed prefixes resolve to the real verb. Only `time`
        // takes a full command list, so env assignments are valid again there
        // (`time FOO=bar cd /tmp` runs the cd); after `command`/`builtin` the
        // next word is the command name and must NOT be skipped.
        i = idx;
        if unquoted == "time" {
            while i < words.len() && super::is_env_assignment(words[i]) {
                i += 1;
            }
        }
    }
}

/// Classify a verb word as provably literal or unprovable.
///
/// A balanced-quoted word (`"cd"`, `'cd'`) is literal (the shell concatenates
/// nothing and the quoted content is the command name). Anything with `$`
/// (variables, `$(...)`, ANSI-C `$'...'`), backslashes, brace-expansion
/// metacharacters (`{touch,}` expands to `touch` — an unmodeled word list),
/// or quote characters outside a balanced pair (`"c"d`, `c'd'`) cannot be
/// normalized without full word-expansion modeling — unprovable, fail-closed.
fn classify_verb_word(w: &str) -> VerbClass<'_> {
    if w.contains(['$', '\\', '{', '}', ',']) {
        return VerbClass::Unprovable;
    }
    if let Some((content, _)) = strip_outer_quotes(w) {
        if content.is_empty() || content.contains(['\'', '"']) {
            return VerbClass::Unprovable;
        }
        return VerbClass::Literal(content);
    }
    if w.contains(['\'', '"']) {
        return VerbClass::Unprovable;
    }
    VerbClass::Literal(w)
}

/// Check a single command segment for unsafe operations.
///
/// `state` carries the tracked current directory and temp-variable bindings
/// across segments (so `cd /tmp && touch f` resolves `f` against `/tmp`).
/// `negated` is set when the segment follows a `!` pipeline-negation
/// operator: `time` loses its reserved-word status there (external command).
#[allow(clippy::too_many_lines)] // security-critical segment validator
fn check_segment(segment: &str, state: &mut ValidationState, negated: bool) -> Result<(), String> {
    let trimmed = segment.trim();
    if trimmed.is_empty() {
        return Ok(());
    }

    // ── CWD tracking ───────────────────────────────────────────────
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
    // forward their command to the current shell (`command`/`builtin`, plus
    // `time` as the unquoted first word — quoted/composed/negated/assigned
    // spellings and a condition/case-arm head (`if time cd /tmp`) resolve to
    // the external `time` and are NOT routed here),
    // with each prefix's option grammar deciding whether the command executes
    // (`command -p`/`--`, `builtin --`, `time -p` forward; `command -v`,
    // `builtin -p`, `time -p -p` are informational). A verb that cannot be
    // normalized to a literal word (`"c"d`, `$'cd'`, `c\144`, `$(...)d`) is
    // unprovable — rejected outright (it could be a cd OR a mutator).
    // `eval` is a verb, not a prefix: its body is evaluated in the current
    // shell, so a `cd` inside is handled like a bare cd. Balanced-quoted verbs
    // (`"cd"`) match by their unquoted content. Non-forwarding prefixes
    // (`sudo`/`env`/`nice`/…) run the command in a child shell, so the real
    // CWD never changes and the segment is not routed here.
    let words: Vec<&str> = split_words_keeping_substitutions(trimmed);
    let (verb_idx, verb) = match resolve_verb(&words, negated) {
        VerbResolution::Informational | VerbResolution::None => {
            // No literal verb here, so the export branch can't fire.
            apply_env_bindings(&words, state, None)?;
            return Ok(());
        }
        VerbResolution::Verb {
            class: VerbClass::Unprovable,
            ..
        } => {
            // A bare command substitution at command position (a standalone
            // `$(...)`/backtick span, e.g. emitted from a heredoc body) is not
            // a real verb: its content executes in a subshell and was already
            // validated by `scan_substitutions`. Only a substitution-FORMED
            // verb with trailing content (`$(printf c)d`) is unprovable.
            if !is_bare_substitution_segment(trimmed) {
                return reject(
                    trimmed,
                    "the command verb cannot be proven safe (concatenated quotes, escapes, or substitution-formed).",
                    "write the command name literally (e.g. `cd`, `rm`) so it can be validated.",
                );
            }
            // Same: unprovable verbs can't be a literal `export` — pass None.
            apply_env_bindings(&words, state, None)?;
            return Ok(());
        }
        VerbResolution::Verb {
            idx,
            class: VerbClass::Literal(v),
        } => (idx, v),
    };
    if matches!(verb, "cd" | "pushd" | "popd") {
        process_cd_words(&words, verb_idx, verb, state);
        return Ok(());
    }
    if verb == "eval" {
        // A cd in the body is handled as a current-shell cd; otherwise fall
        // through so the body is validated as its own command (first_command_word
        // strips `eval`), matching pre-existing behavior for eval write content.
        match handle_eval_body(&words[verb_idx + 1..], state) {
            Ok(true) => return Ok(()),
            Ok(false) => {}
            Err(e) => return Err(e),
        }
    }
    // NOTE: a `{` at verb position never reaches here — `scan_list` parses
    // real brace groups at command position (`parse_brace`), and any other
    // `{`-containing verb is classified Unprovable above and rejected.

    // ── Temp-variable bindings ─────────────────────────────────────
    // `export X=...`, plain `X=...` segments, and env-prefix forms
    // (`TMPDIR=/tmp cmd`) bind the temp variables; a non-temp binding poisons
    // the variable, a temp-root binding clears the poison.
    //
    // The export verb is the FULL word-list resolution (not the assignment-
    // stripped slice): `time` is a forwarding prefix only at position 0, so
    // `TMPDIR=/tmp time export X=...` resolves the external `time` and must
    // not bind the export — a slice-relative head would wrongly bind it.
    // Only this Literal-verb call site can fire that branch; the two earlier
    // arms pass None.
    apply_env_bindings(&words, state, Some((verb_idx, verb)))?;

    // Extract the effective command by stripping shell prefixes and
    // environment variable assignments.
    let first_word = super::first_command_word(trimmed);

    if first_word.is_empty() {
        return Ok(());
    }

    // Normalize the verb for blocklist matching: a balanced-quoted word
    // (`"rm"` executes `rm`) matches by its unquoted content; a concatenated/
    // ANSI-C/escape/substitution-formed verb cannot be proven — reject rather
    // than let it bypass the mutator/git/cargo/flag checks (fail-closed).
    let first_word = match classify_verb_word(first_word) {
        VerbClass::Literal(v) => v,
        VerbClass::Unprovable => {
            return reject(
                trimmed,
                "the command verb cannot be proven safe (concatenated quotes, escapes, or substitution-formed).",
                "write the command name literally (e.g. `rm`, `touch`) so it can be validated.",
            );
        }
    };

    // 'mktemp' creates a temp directory and outputs its path — always allowed.
    if first_word == "mktemp" {
        return Ok(());
    }

    // Temp-gated mutator dispatch (scratch, temp, unconditional). A matching
    // verb is rejected when its reject predicate reports a write outside temp
    // (`None` = always rejected). `mkdir` additionally records its provably
    // created directories (all `-p` targets, and non-`-p` targets whose parent
    // already exists) so a later `cd` in the same chain can track them even
    // though they did not exist at validation time — a non-`-p` target under a
    // missing parent errors at runtime and must not be treated as created
    // (fail-closed).
    for check in MUTATOR_CHECKS {
        if !check.verbs.contains(&first_word) {
            continue;
        }
        if check
            .rejects
            .is_none_or(|reject| reject(trimmed, first_word, state))
        {
            let (education, fallback) = check.suggestions;
            return reject(
                trimmed,
                &check.rejection.replace("{verb}", first_word),
                if has_unresolved_var_path(trimmed, state) {
                    education
                } else {
                    fallback
                },
            );
        }
        if first_word == "mkdir" {
            record_mkdir_targets(trimmed, state);
        }
        return Ok(());
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

/// Resolve the target of a `cd`/`pushd`/`popd` verb at `cd_idx` (shared by
/// direct cd segments and decoded `eval` bodies). pushd/popd always reset
/// fail-closed; only `cd` tracks.
///
/// Skip option words before resolving the real target. `-P`/`-L` (and
/// combined forms) are valid; `--` ends option parsing; any other `-…`
/// option (`-e`, `-@`, `-x`, …) errors at runtime, so the `cd` never
/// happens and the real CWD stays put — tracking would approve a chained
/// write that lands there. A bare `cd`/`cd -P` (no target after the flags)
/// targets `$HOME`. All of these reset fail-closed; `-` alone is the
/// `$OLDPWD` target, never an option. Option detection uses the raw word
/// (a quoted `"-P"` is a literal target), while the resolved target is
/// quote-stripped (`cd "/tmp"`).
fn process_cd_words(words: &[&str], cd_idx: usize, verb: &str, state: &mut ValidationState) {
    // Every executed cd/pushd/popd moves the real CWD in the current shell —
    // construct parsers compare this counter to detect the leak.
    state.cd_count += 1;
    if verb != "cd" {
        state.cwd = None;
        return;
    }
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
        break Some(strip_outer_quotes(w).map_or(*w, |(c, _)| c));
    };
    let Some(target) = target else {
        state.cwd = None;
        return;
    };
    // A word after the target is an extra operand (`cd /tmp extra`). The
    // runtime shell (bash 3.2) ignores extras and executes the cd to the first
    // target; the guard cannot prove the target across shells — reset
    // fail-closed.
    if words.get(i + 1).is_some() {
        state.cwd = None;
        return;
    }
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
            // Filesystem existence checks are expected and acceptable;
            // a nonexistent target fails the command, so
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

/// An `eval` segment evaluates its body in the current shell, so a `cd` in
/// the body moves the real CWD behind the guard's back. The body is decoded
/// (one layer of surrounding quotes) and processed like a bare cd when it is
/// a clean simple `cd [target]`; any other cd/pushd/popd shape — extra
/// commands, separators, mixed quoting, or a pushd/popd — resets tracking
/// fail-closed. A body whose first command verb cannot be normalized to a
/// literal word (`eval '"c"d /etc'`) could be a current-shell cd or a mutator
/// — rejected outright. A fully quoted body without a cd verb is the
/// documented eval-write residual (accepted — the guard does not model its
/// content). Returns `Ok(true)` when the segment was consumed; a body with no
/// cd verb that is not fully quoted returns `Ok(false)` so the caller
/// validates it as its own command.
fn handle_eval_body(body_words: &[&str], state: &mut ValidationState) -> Result<bool, String> {
    let joined = body_words.join(" ");
    let (decoded, clean, quoted) = if let Some((content, _)) = strip_outer_quotes(&joined) {
        (content.to_string(), true, joined.starts_with(['\'', '"']))
    } else {
        let mut s = String::with_capacity(joined.len());
        s.extend(joined.chars().filter(|c| !matches!(c, '\'' | '"')));
        (s, false, false)
    };
    let toks: Vec<&str> = decoded.split_whitespace().collect();
    if let Some(first) = toks.first()
        && matches!(classify_verb_word(first), VerbClass::Unprovable)
    {
        return reject(
            &joined,
            "the eval body's command verb cannot be proven safe (concatenated quotes, escapes, or substitution-formed).",
            "write the command name literally (e.g. `cd`, `rm`) so it can be validated.",
        );
    }
    let Some((ti, tv)) = toks.iter().enumerate().find_map(|(i, w)| {
        let u = strip_outer_quotes(w).map_or(*w, |(c, _)| c);
        matches!(u, "cd" | "pushd" | "popd").then_some((i, u))
    }) else {
        // No cd verb. A fully quoted body is the documented eval-write
        // residual — accepted by design. An unquoted body is visible and
        // falls through for normal validation.
        return Ok(quoted);
    };
    let simple = clean
        && tv == "cd"
        && ti == 0
        && !toks[1..].iter().any(|w| {
            let u = strip_outer_quotes(w).map_or(*w, |(c, _)| c);
            matches!(u, "&&" | "||" | ";" | "|" | "cd" | "pushd" | "popd")
                || u.ends_with([';', '&', '|'])
        });
    if !simple {
        state.cd_count += 1;
        state.cwd = None;
        return Ok(true);
    }
    process_cd_words(&toks, 0, "cd", state);
    Ok(true)
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

/// Apply temp-variable bindings from a segment: `export X=...`,
/// plain `X=...` assignment segments, and leading env-prefix assignments
/// (`TMPDIR=/tmp cmd`). Bare `export` / `export NAME` are no-ops.
///
/// The `export` verb is located with the same normalization as the rest of
/// the guard: quoted spellings (`"export" X=...`, `'export' X=...`) and the
/// forwarding prefixes that run in the current shell (`command export X=...`,
/// `time export ...` — time only as the unquoted first word outside
/// condition heads) bind like a plain export. The verb search runs against the FULL word list: an env
/// assignment before `time` (`TMPDIR=/tmp time export X=...`) demotes it to
/// the external command, whose export never reaches the parent — nothing
/// binds. Assignment words are quote-normalized too (`export "TMPDIR=/etc"`,
/// `"TMPDIR"=/etc` bind by their unquoted name). Non-forwarding prefixes
/// (`env export`, `sudo export`) run in a child shell — the export never
/// reaches the parent, so nothing binds there.
/// Reject `GIT_*` env bindings (quote-stripped) except the documented
/// `GIT_PAGER` carve-out — git only paginates on a TTY and the shell tool
/// captures via pipes, so a pager binding can never spawn. Covers
/// GIT_EXTERNAL_DIFF, GIT_SSH_COMMAND, GIT_CONFIG_*, GIT_DIR, GIT_EXEC_PATH,
/// GIT_TRACE*, GIT_ASKPASS — all invisible to the subcommand allowlist.
/// Also fires on non-git commands (`GIT_DIR=/tmp ls`): fail-closed trade-off
/// closing transitive git invocation (make/cargo inheriting GIT_*).
fn check_git_env_binding(word: &str) -> Result<(), String> {
    let w = strip_outer_quotes(word).map_or(word, |(c, _)| c);
    if let Some((name, _)) = w.split_once('=')
        && name.starts_with("GIT_")
        && name != "GIT_PAGER"
    {
        return reject(
            word,
            &format!(
                "`{name}` environment bindings are not allowed — git env vars can execute programs (GIT_EXTERNAL_DIFF, GIT_SSH_COMMAND, ...)."
            ),
            "run git without GIT_* environment assignments.",
        );
    }
    Ok(())
}

fn apply_env_bindings(
    words: &[&str],
    state: &mut ValidationState,
    export_verb: Option<(usize, &str)>,
) -> Result<(), String> {
    let first_non_assign = words
        .iter()
        .position(|w| !super::is_env_assignment(w))
        .unwrap_or(words.len());
    for w in &words[..first_non_assign] {
        check_git_env_binding(w)?;
        bind_assignment_word(w, state);
    }
    if let Some((idx, v)) = export_verb
        && v == "export"
    {
        for w in &words[idx + 1..] {
            check_git_env_binding(w)?;
            bind_assignment_word(w, state);
        }
    }
    Ok(())
}

/// Bind a single `NAME=value` assignment word, stripping balanced surrounding
/// quotes first so `export "TMPDIR=/etc"` and `"TMPDIR"=/etc` bind by their
/// unquoted name (the shell concatenates quoted pieces into one word).
fn bind_assignment_word(w: &str, state: &mut ValidationState) {
    let w = strip_outer_quotes(w).map_or(w, |(c, _)| c);
    if let Some((name, value)) = w.split_once('=') {
        apply_single_binding(name, value, state);
    }
}

/// Split a segment into whitespace-separated words, keeping `$(...)` /
/// backtick substitutions whole — their bodies may contain spaces, and
/// `NAME=$(mktemp -d)` must stay one assignment word.
fn split_words_keeping_substitutions(s: &str) -> Vec<&str> {
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
        if !escaped
            && !in_single
            && let Some((_, next)) = substitution_span(s, i)
        {
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
    // Quoted assignment names (`"TMPDIR"=/etc`) bind by their unquoted content.
    let name = strip_outer_quotes(name).map_or(name, |(c, _)| c);
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

/// Classification of a single `git branch`/`git tag` option (tables verified
/// against git 2.50.1). Unknown/ambiguous long prefixes and unknown short
/// flags fail closed: they are treated as [`RefOpt::Plain`] (create-active),
/// so a bare word after them is a ref name and rejects.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RefOpt {
    /// No value; does not trigger list mode. A bare word after it is a ref name.
    Plain,
    /// Optional `=value` only — a space-separated value is NOT consumed and
    /// becomes a bare ref name. Does not trigger list mode.
    OptVal,
    /// Required value — consumes the next token unconditionally (even when
    /// dash-prefixed). Does not trigger list mode.
    Val,
    /// List-mode trigger — positionals become patterns (no name check).
    List,
    /// Required value + list-mode trigger.
    ListVal,
    /// List-mode trigger with an OPTIONAL attached value (`tag -n5`); a
    /// space value is NOT consumed.
    ListOptVal,
    /// Tag verify mode — positionals are read-only verify targets.
    Verify,
    /// Mutates a ref — rejected regardless of position or list mode.
    Mutation,
}

/// Shared `git branch`/`git tag` long options (git 2.50.1). Negated forms of
/// value-taking options consume no value (verified: `--no-sort foo` creates
/// `foo`), so they classify as [`RefOpt::Plain`] except the real
/// `--no-merged`/`--no-contains` flags. Emits ONE array per tool — resolving
/// the common rows separately would change prefix-abbreviation (`git branch
/// --no-m` must stay ambiguous via `--no-merged` + `--no-move`).
macro_rules! git_ref_opts {
    ($($extra:tt)*) => { &[
        ("--color", RefOpt::OptVal),
        ("--no-color", RefOpt::Plain),
        ("--contains", RefOpt::ListVal),
        ("--no-contains", RefOpt::ListVal),
        ("--delete", RefOpt::Mutation),
        ("--omit-empty", RefOpt::Plain),
        ("--no-omit-empty", RefOpt::Plain),
        ("--list", RefOpt::List),
        ("--create-reflog", RefOpt::Plain),
        ("--no-create-reflog", RefOpt::Plain),
        ("--force", RefOpt::Mutation),
        ("--no-force", RefOpt::Plain),
        ("--merged", RefOpt::ListVal),
        ("--no-merged", RefOpt::ListVal),
        ("--column", RefOpt::OptVal),
        ("--no-column", RefOpt::Plain),
        ("--sort", RefOpt::Val),
        ("--no-sort", RefOpt::Plain),
        ("--points-at", RefOpt::ListVal),
        ("--no-points-at", RefOpt::Plain),
        ("--ignore-case", RefOpt::Plain),
        ("--no-ignore-case", RefOpt::Plain),
        ("--format", RefOpt::Val),
        ("--no-format", RefOpt::Plain),
        $($extra)*
    ] };
}

/// `git branch` long options (git 2.50.1). The deprecated `--set-upstream`
/// alias is covered by prefix-abbreviation to `--set-upstream-to` (git rejects
/// it at runtime).
const GIT_BRANCH_OPTS: &[(&str, RefOpt)] = git_ref_opts![
    ("--verbose", RefOpt::Plain),
    ("--no-verbose", RefOpt::Plain),
    ("--quiet", RefOpt::Plain),
    ("--no-quiet", RefOpt::Plain),
    ("--track", RefOpt::Mutation),
    ("--no-track", RefOpt::Plain),
    ("--set-upstream-to", RefOpt::Mutation),
    ("--no-set-upstream-to", RefOpt::Plain),
    ("--unset-upstream", RefOpt::Mutation),
    ("--no-unset-upstream", RefOpt::Plain),
    ("--remotes", RefOpt::Plain),
    ("--abbrev", RefOpt::OptVal),
    ("--no-abbrev", RefOpt::Plain),
    ("--all", RefOpt::Plain),
    ("--no-delete", RefOpt::Plain),
    ("--move", RefOpt::Mutation),
    ("--no-move", RefOpt::Plain),
    ("--copy", RefOpt::Mutation),
    ("--no-copy", RefOpt::Plain),
    ("--no-list", RefOpt::Plain),
    ("--show-current", RefOpt::Plain),
    ("--no-show-current", RefOpt::Plain),
    ("--edit-description", RefOpt::Mutation),
    ("--no-edit-description", RefOpt::Plain),
    ("--recurse-submodules", RefOpt::Plain),
    ("--no-recurse-submodules", RefOpt::Plain),
];

/// `git branch` short flags (git 2.50.1).
const GIT_BRANCH_SHORTS: &[(char, RefOpt)] = &[
    ('v', RefOpt::Plain),    // verbose — create-active
    ('q', RefOpt::Plain),    // quiet
    ('t', RefOpt::Mutation), // track
    ('u', RefOpt::Mutation), // set-upstream-to
    ('r', RefOpt::Plain),    // remotes — list-only, create-active with a name
    ('a', RefOpt::Plain),    // all
    ('d', RefOpt::Mutation),
    ('D', RefOpt::Mutation),
    ('m', RefOpt::Mutation),
    ('M', RefOpt::Mutation),
    ('c', RefOpt::Mutation),
    ('C', RefOpt::Mutation),
    ('l', RefOpt::List),
    ('f', RefOpt::Mutation),
    ('i', RefOpt::Plain), // ignore-case
];

/// `git tag` long options (git 2.50.1). `--no-message`/`--no-list` are unknown
/// in git (fail-closed Plain); every accepted `--no-*` negation consumes no
/// value and is create-active.
const GIT_TAG_OPTS: &[(&str, RefOpt)] = git_ref_opts![
    ("--verify", RefOpt::Verify),
    ("--annotate", RefOpt::Mutation),
    ("--no-annotate", RefOpt::Plain),
    ("--message", RefOpt::Mutation),
    ("--file", RefOpt::Mutation),
    ("--no-file", RefOpt::Plain),
    ("--trailer", RefOpt::Mutation),
    ("--edit", RefOpt::Mutation),
    ("--no-edit", RefOpt::Plain),
    ("--sign", RefOpt::Mutation),
    ("--no-sign", RefOpt::Plain),
    ("--cleanup", RefOpt::Mutation),
    ("--no-cleanup", RefOpt::Plain),
    ("--local-user", RefOpt::Mutation),
    ("--no-local-user", RefOpt::Plain),
];

// Row-count guard: dropped/duplicated rows change these totals (the table
// tests only assert Mutation-class rows).
const _: () = assert!(GIT_BRANCH_OPTS.len() == 50 && GIT_TAG_OPTS.len() == 39);

/// `git tag` short flags (git 2.50.1).
const GIT_TAG_SHORTS: &[(char, RefOpt)] = &[
    ('l', RefOpt::List),
    ('n', RefOpt::ListOptVal),
    ('d', RefOpt::Mutation),
    ('v', RefOpt::Verify),
    ('a', RefOpt::Mutation),
    ('m', RefOpt::Mutation),
    ('F', RefOpt::Mutation),
    ('e', RefOpt::Mutation),
    ('s', RefOpt::Mutation),
    ('u', RefOpt::Mutation),
    ('f', RefOpt::Mutation),
    ('i', RefOpt::Plain),
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

/// Git subcommands that always write to the object database or working tree —
/// rejected regardless of flags. Prefix-matched (`subcommand.starts_with`).
const GIT_ALWAYS_MUTATE: &[(&str, &str, &str)] = &[
    // git mktag always writes a tag object to the object database — no read-only mode.
    (
        "mktag",
        "`git mktag` is not allowed — it always writes a tag object to the object database.",
        "use `git verify-tag` or `git cat-file` to inspect existing tag objects.",
    ),
    // git mktree always writes a tree object to the object database — no read-only mode.
    (
        "mktree",
        "`git mktree` is not allowed — it always writes a tree object to the object database.",
        "use `git ls-tree` to inspect existing tree objects.",
    ),
    // git merge-file without -p/--stdout overwrites the <current> file in-place.
    // Even with -p/--stdout, the --object-id variant writes to the object store.
    // Rather than a complex multi-flag check, reject entirely for safety.
    (
        "merge-file",
        "`git merge-file` is not allowed — it mutates files or writes to the object database.",
        "use `git diff` to compare files, or `diff`/`diff3` for three-way comparisons.",
    ),
    // git merge-tree since Git 2.40 defaults to --write-tree, which creates tree
    // objects in the object database. The alternative --trivial-merge is deprecated.
    (
        "merge-tree",
        "`git merge-tree` is not allowed — it writes tree objects in its default mode.",
        "use `git merge-base` to find the merge base, or `git diff-tree` to inspect trees.",
    ),
];

/// Value-aware dry-run/force scan for `git push`/`git clean`. The caller
/// passes the value-taking options: push `-o`/`--push-option` and `--repo`,
/// clean `-e`/`--exclude`. A value-taking option consumes dry-run/force-looking
/// tokens as its value (`git push -o --dry-run`, `git push --repo --dry-run`
/// are real pushes), so the scan skips those values: attached (`-onope`),
/// `=`-form (`-o=n`, `--repo=--dry-run`), and the token following a bare
/// value-taking option, even dash-leading (`-o -n`, `--exclude --dry-run`).
/// Long dry-run/force tokens match exactly on the raw token — normalizing
/// would let a quoted value masquerade as a flag (`-o "--dry-run"`) — while
/// short clusters are scanned on the shell-normalized token (`'-o'` is
/// value-taking). `-u` is boolean `--set-upstream`, never value-taking
/// (`-nu`/`-un` stay dry-runs). Long prefixes of the value-taking options are
/// value-taking too (git abbreviates; ambiguous prefixes fail closed). A
/// standalone `--` (quote-normalized) ends the scan; a `--` consumed as an
/// option value does not. Returns `(has_dry_run, has_force)`.
fn scan_push_clean_tokens(
    subcommand: &str,
    value_short: char,
    value_longs: &[&str],
) -> (bool, bool) {
    let words = split_words_keeping_substitutions(subcommand);
    let mut dry = false;
    let mut force = false;
    let mut i = 1; // words[0] is the subcommand name
    while i < words.len() {
        let raw = words[i];
        let wq = shell_word(raw);
        if wq == "--" {
            break; // path separator; a consumed `--` never reaches here
        }
        if wq.starts_with("--") {
            if raw == "--dry-run" {
                dry = true;
            } else if raw == "--force" {
                force = true;
            } else if value_longs
                .iter()
                .any(|l| l.starts_with(wq.split('=').next().unwrap_or(&wq)))
            {
                // `=`-form carries its value; bare form consumes the next token
                if !wq.contains('=') {
                    i += 1;
                }
            }
        } else if wq.starts_with('-') && wq.len() > 1 {
            let b = wq.as_bytes();
            let mut k = 1;
            while k < b.len() {
                let c = b[k] as char;
                if c == 'n' {
                    dry = true;
                } else if c == 'f' {
                    force = true;
                } else if c == value_short {
                    // rest of token is the attached value; a trailing
                    // value_short consumes the next token instead
                    if k + 1 < b.len() {
                        break;
                    }
                    i += 1;
                    break;
                }
                k += 1;
            }
        }
        i += 1;
    }
    (dry, force)
}

/// Phase-3 read-only git allowlist rules: `stash show`, `config` read forms,
/// `rebase --show-current`, `push --dry-run`/`-n`, `clean -n`/`--dry-run`,
/// and `submodule status`.
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

    // git config read/write rule: exactly one positional after
    // `config` (key read) is allowed; two positionals (key + value) write.
    // Explicit get forms (--list, -l, --get, --get-all, --get-regexp,
    // --name-only) are allowed; write/edit forms are blocked.
    if subcommand.starts_with("config") {
        let words: Vec<&str> = subcommand.split_whitespace().collect();
        let rest = &words[1..];
        if rest.iter().any(|w| {
            let w = shell_word(w);
            matches!(
                w.as_str(),
                "--list" | "-l" | "--get" | "--get-all" | "--get-regexp" | "--name-only"
            )
        }) {
            return Some(Ok(()));
        }
        if rest.iter().any(|w| {
            let w = shell_word(w);
            matches!(
                w.as_str(),
                "--add"
                    | "--unset"
                    | "--unset-all"
                    | "--edit"
                    | "--remove-section"
                    | "--rename-section"
                    | "--replace-all"
            )
        }) || has_cluster_char(&rest.join(" "), &['e'], &['f'])
        {
            // Long write/edit forms match after shell normalization (quoted
            // `--'edit'` delivers `--edit`); the short `-e` (edit) form matches
            // inside combined clusters with `-f` value-taking arity
            // (`-fe` is `-f e`, not edit).
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
        // local mutation; any force token in the same command blocks it.
        // Value-aware: `-o`/`--push-option` and `--repo`
        // consume dry-run/force-looking values, so a real push can't
        // masquerade. (Other push value-takers self-block: `--recurse-
        // submodules` validates its value, `--signed`/`--force-with-lease`
        // take optional `=`-only args; `--exec`/`--receive-pack` are closed
        // by the exec-vector layer.)
        let (dry, force) = scan_push_clean_tokens(subcommand, 'o', &["--push-option", "--repo"]);
        if dry && !force {
            Some(Ok(()))
        } else {
            Some(reject(
                trimmed,
                "`git push` is not allowed — it writes to a remote repository.",
                "use `git push --dry-run` to preview what would be pushed without sending anything.",
            ))
        }
    } else if subcommand.starts_with("clean") {
        // git clean -n/--dry-run previews removals; any force token blocks it.
        // Value-aware: `-e`/`--exclude` consumes
        // dry-run/force-looking values, so a real clean can't masquerade.
        let (dry, force) = scan_push_clean_tokens(subcommand, 'e', &["--exclude"]);
        if dry && !force {
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

/// Reject git invocations that can execute programs via flag/config/env
/// channels invisible to the subcommand allowlist. Runs FIRST in
/// [`check_git_segment`] — before [`check_git_read_only_extensions`], whose
/// allowlisted forms (`git stash show --textconv`, `git push --dry-run
/// --exec=`) would otherwise bypass the scan. Fail-closed: repo-redirect
/// globals (`-C`/`--git-dir`/`--work-tree`) are rejected too, so an
/// agent-written hostile repo config cannot be reached via redirect.
/// Accepted residual: a repo with pre-existing hostile config (`.git/config`
/// diff.external/filter.*/core.fsmonitor, .gitattributes textconv, global
/// LFS filters) can still exec drivers from plain `git diff`/`status` —
/// trusted-workspace model, out of scope here.
fn check_git_exec_vectors(trimmed: &str, subcommand: &str) -> Result<(), String> {
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let Some(git_idx) = super::find_first_command_word_index(&words) else {
        return Ok(());
    };
    // Env bindings before `git` — catches `env`/`sudo`-wrapped forms that the
    // binding layer (which only sees current-shell segments) cannot.
    for w in &words[..git_idx] {
        if super::is_env_assignment(w) {
            check_git_env_binding(w)?;
        }
    }
    // Global-region flags between `git` and the subcommand. Position-sensitive:
    // `git log -c` / `git diff -c` (combined diff) live in the subcommand
    // region and stay allowed. `-c` injects config that executes programs;
    // `-C` redirects the repo (attached/clustered forms `-cfoo`, `-pc`, `-pC`
    // included). Long globals redirect the repo or inject config. The
    // subcommand start is derived with the same shared helper
    // `extract_git_subcommand` uses, so the global region always ends where
    // the subcommand begins.
    let sub_idx = super::find_first_non_flag_index(&words[git_idx + 1..], true)
        .map_or(words.len(), |i| git_idx + 1 + i);
    for w in &words[git_idx + 1..sub_idx] {
        let wq = shell_word(w);
        if wq.starts_with('-') && !wq.starts_with("--") && wq[1..].contains(['c', 'C']) {
            return reject(
                trimmed,
                "`-c`/`-C` git global options are not allowed in read-only mode — they inject config or redirect the repository, both of which can execute programs.",
                "run git without global `-c`/`-C` options (use `cd` to change directory).",
            );
        }
        for opt in [
            "--git-dir",
            "--work-tree",
            "--config-env",
            "--config-file",
            "--exec-path",
        ] {
            if wq == opt || wq.starts_with(&format!("{opt}=")) {
                return reject(
                    trimmed,
                    &format!(
                        "`{opt}` is not allowed in read-only mode — it redirects the repository or injects config, enabling program execution."
                    ),
                    "run git against the workspace repository without repo-redirect or config-injection options.",
                );
            }
        }
    }
    check_git_exec_flags(trimmed, subcommand, &words)?;
    check_git_scoped_flags(trimmed, subcommand, &words)
}

/// Exec-flag blocks applied across the whole command. Git abbreviates
/// unambiguous long-option prefixes (`--upload-p=` == `--upload-pack=`), so
/// any token of length > 2 that is a prefix of a blocked flag is rejected
/// (bare `--` is the path separator, never an option). Exact `--filter` and
/// `--text` are exempted where git resolves them to benign flags — `--filter`
/// shadows `--filters`; `--text` is the `-a` flag on the diff family + grep
/// (per `GIT_TEXT_BENIGN_SUBCOMMANDS`), while elsewhere git abbreviates it to
/// `--textconv` (e.g. `git cat-file --text` execs the driver), so it stays
/// blocked there. `--no-*` disabling forms stay allowed. Fail-closed
/// over-rejections: pathnames literally named like a blocked flag
/// (`git log -- --ext-diff`) and literal grep patterns (`git grep -e
/// '--ext-diff'`). ANSI-C `$'...\...'` spans (standalone or embedded like
/// `--format=$'%h\t%s'`) are unprovable flag candidates — same contract as
/// [`is_unprovable_flag_token`].
const GIT_EXEC_LONG_FLAGS: &[&str] = &[
    "--ext-diff",
    "--textconv",
    "--show-signature",
    "--filters",
    "--upload-pack",
    "--exec",
    "--receive-pack",
];

/// Subcommands where git resolves exact `--text` to the benign `-a/--text`
/// flag (diff family, grep, and shortlog/rev-list which take diff options).
/// On all other safe subcommands `--text` is either invalid or — like
/// `cat-file` — abbreviates to `--textconv`, an exec vector.
const GIT_TEXT_BENIGN_SUBCOMMANDS: &[&str] = &[
    "grep",
    "diff",
    "log",
    "show",
    "blame",
    "annotate",
    "diff-files",
    "diff-index",
    "diff-tree",
    "whatchanged",
    "shortlog",
    "rev-list",
    "range-diff",
    "stash",
];

fn check_git_exec_flags(trimmed: &str, subcommand: &str, words: &[&str]) -> Result<(), String> {
    let base = subcommand.split_whitespace().next().unwrap_or("");
    for w in words {
        if w.contains("$'") && w.contains('\\') {
            return reject(
                trimmed,
                "ANSI-C quoted git arguments with backslash escapes cannot be proven safe.",
                "write git flags and arguments literally.",
            );
        }
        let wq = shell_word(w);
        let opt = wq.split('=').next().unwrap_or(&wq);
        // Bare `-`/`--` are never options — `--` is the path separator and git
        // abbreviates only long options of the form `--x...`.
        if opt.len() <= 2 {
            continue;
        }
        // `git grep -e '--ext-diff'` (literal pattern) is over-rejected too —
        // accepted fail-closed asymmetry with the spared `--regexp=` form.
        let text_benign = opt == "--text" && GIT_TEXT_BENIGN_SUBCOMMANDS.contains(&base);
        if opt != "--filter"
            && !text_benign
            && GIT_EXEC_LONG_FLAGS.iter().any(|f| f.starts_with(opt))
        {
            return reject(
                trimmed,
                &format!(
                    "`{opt}` is not allowed in read-only mode — it selects a git feature that executes external programs."
                ),
                "drop the flag; the corresponding git feature stays disabled without it.",
            );
        }
    }
    Ok(())
}

/// Subcommand-scoped exec flags. `-O`/`--open-files-in-pager` (grep only:
/// runs the program per matched file — `git log -O<orderfile>` is a benign
/// file read, `grep -o`/`-c`/`-C3` are only-matching) and the help viewers
/// `-w/--web`, `-m/--man`, `-i/--info` (each execs an external program —
/// `git diff -w` is ignore-whitespace). Long-option prefixes are blocked
/// (git abbreviates; `--open-files-in-p=` == `--open-files-in-pager=`);
/// short-flag matches are case-sensitive: uppercase `O`, lowercase `w`/`m`/`i`.
fn check_git_scoped_flags(trimmed: &str, subcommand: &str, words: &[&str]) -> Result<(), String> {
    let base = subcommand.split_whitespace().next().unwrap_or("");
    for w in words {
        let wq = shell_word(w);
        let opt = wq.split('=').next().unwrap_or(&wq);
        let is_scoped = match base {
            "grep" => {
                (opt.len() > 2 && "--open-files-in-pager".starts_with(opt))
                    || (wq.starts_with('-') && !wq.starts_with("--") && wq[1..].contains('O'))
            }
            "help" => {
                (opt.len() > 2
                    && ["--web", "--man", "--info"]
                        .iter()
                        .any(|f| f.starts_with(opt)))
                    || (wq.starts_with('-')
                        && !wq.starts_with("--")
                        && wq[1..].contains(['w', 'm', 'i']))
            }
            _ => false,
        };
        if is_scoped {
            return reject(
                trimmed,
                &format!(
                    "`{wq}` on `git {base}` is not allowed in read-only mode — it runs an external program."
                ),
                "drop the flag.",
            );
        }
    }
    Ok(())
}

/// Reject `--output` on any git subcommand — it writes the command's output
/// to a file (`--output=<file>` / `--output <file>`; blame/annotate/rev-list
/// truncate the target to 0 bytes). Runs before
/// [`check_git_read_only_extensions`], whose read forms (`git stash show
/// --output=...`) would otherwise bypass it. Exact token (not [`flag_value`]:
/// git takes dash filenames, `git diff --output -1`); `--output-indicator-*`
/// do not write. Fail-closed over-rejection: `--output` as a value of
/// `-S`/`-G`/`-e` (`git log -S --output`) is a literal read-only search.
fn check_git_output_flag(trimmed: &str, subcommand: &str) -> Result<(), String> {
    if subcommand.split_whitespace().any(|w| {
        let w = shell_word(w);
        w == "--output" || w.starts_with("--output=")
    }) {
        return reject(
            trimmed,
            "`--output` is not allowed in read-only mode — it writes the git output to a file.",
            "drop the flag; use a shell redirect like `git diff > /tmp/out` to save output to the OS temp directory.",
        );
    }
    Ok(())
}

fn check_git_segment(segment: &str) -> Result<(), String> {
    let trimmed = segment.trim();

    // Extract the git subcommand by skipping "git" and global flags
    let subcommand = extract_git_subcommand(trimmed);

    // Exec-vector scan FIRST — before the extension allowlist and the
    // empty-subcommand early return (`git --git-dir=/tmp/evil` bare still
    // rejects the repo redirect).
    check_git_exec_vectors(trimmed, &subcommand)?;

    if subcommand.is_empty() || subcommand == "git" {
        return Ok(());
    }

    // `--output` file-write vector — before the extension allowlist (stash read forms bypass it).
    check_git_output_flag(trimmed, &subcommand)?;

    // Phase-3 read-only allowlist rules (stash/config/rebase/push/clean/
    // submodule). These run BEFORE the general safe-list match because the
    // subcommands are not (all) in GIT_SAFE_SUBCOMMANDS, and the stash check
    // must exempt `stash show` at this layer.
    if let Some(result) = check_git_read_only_extensions(trimmed, &subcommand) {
        return result;
    }

    // git subcommands that always write to the object database or working
    // tree — rejected regardless of flags (prefix-matched).
    for &(prefix, why, suggestion) in GIT_ALWAYS_MUTATE {
        if subcommand.starts_with(prefix) {
            return reject(trimmed, why, suggestion);
        }
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

    // Additional mutation-flag checks for branch/tag/remote/hash-object/reflog/fsck
    match matched_safe {
        Some("branch") => check_git_ref_subcommand(&subcommand, "branch")?,
        Some("tag") => check_git_ref_subcommand(&subcommand, "tag")?,
        Some("remote") => {
            check_git_subcommand_mutation(&subcommand, GIT_REMOTE_MUTATIONS)?;
        }
        Some("hash-object") => {
            // git hash-object -w writes the object to the database; without -w
            // it only computes and outputs the hash (read-only). The -w flag can
            // appear anywhere in the argument list, alone or in a combined
            // cluster (`-wt blob <file>`). `-t` is value-taking and consumes the
            // rest of its token (`-tw` is `-t w`, not `-w`). `-wt <file>` (no
            // type) is over-rejected — `-t` would consume the file, so no write
            // happens — an accepted fail-closed contract.
            if has_cluster_char(trimmed, &['w'], &['t']) {
                return reject(
                    trimmed,
                    "`git hash-object` with `-w` is not allowed — it writes objects to the object database.",
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
        // `git fsck --lost-found` writes dangling objects into `.git/lost-found/`.
        // Git abbreviates unambiguous long options, so any token starting with
        // `--l` triggers it (no other fsck long option starts with `l`);
        // `--no-lost-found` (starts with `--n`) disables the write and stays
        // allowed. shell_word strips quotes/backslashes, closing the
        // quoted/escaped forms.
        Some("fsck")
            if subcommand
                .split_whitespace()
                .any(|w| shell_word(w).starts_with("--l")) =>
        {
            return reject(
                trimmed,
                "`git fsck --lost-found` is not allowed — it writes dangling objects to `.git/lost-found/`.",
                "drop the flag; dangling objects are still listed on stdout without `--lost-found`.",
            );
        }
        _ => {}
    }

    Ok(())
}

/// True when `word` is a mutation token or the `token=value` form of one.
fn matches_mutation_token(word: &str, tokens: &[&str]) -> bool {
    tokens.contains(&word)
        || tokens
            .iter()
            .any(|t| word.strip_prefix(t).is_some_and(|r| r.starts_with('=')))
}

/// Resolve a long branch/tag option token (with optional `=value`) against
/// `table`, honoring git's unambiguous-prefix abbreviation. Returns `None`
/// for unknown or ambiguous prefixes (fail closed — create-active).
fn resolve_ref_long_opt(token: &str, table: &[(&str, RefOpt)]) -> Option<RefOpt> {
    let base = token.split('=').next().unwrap_or(token);
    let mut matches = table.iter().filter(|(name, _)| name.starts_with(base));
    let first = matches.next()?;
    if matches.next().is_some() {
        return None; // ambiguous abbreviation — git errors; fail closed
    }
    Some(first.1)
}

/// True when `word` (shell-normalized) is a branch/tag mutation token: a
/// long option resolving to [`RefOpt::Mutation`] (exact, `=value`, or
/// unambiguous abbreviation), or a single-dash cluster containing a mutation
/// short char. Runs on every token regardless of option-value consumption
/// (`git branch --merged -d` must still reject; `git branch --sort -d foo`
/// likewise).
fn is_ref_mutation_word(word: &str, table: &[(&str, RefOpt)], shorts: &[(char, RefOpt)]) -> bool {
    if let Some(stripped) = word.strip_prefix("--") {
        return !stripped.is_empty() && resolve_ref_long_opt(word, table) == Some(RefOpt::Mutation);
    }
    if word.len() > 1 && word.starts_with('-') {
        return shorts
            .iter()
            .any(|(c, kind)| *kind == RefOpt::Mutation && word[1..].contains(*c));
    }
    false
}

fn git_mutation_rejection(subcommand: &str) -> String {
    format!(
        "⚠️ Read-only mode: `git {subcommand}` is not allowed — it mutates.\n\
         Suggestion: inspect state with read-only git commands instead \
         (e.g. `git status`, `git log`, `git diff`, `git show`)."
    )
}

/// Check a `git branch`/`git tag` subcommand against the git 2.50.1 option
/// tables for ref mutations and ref-name creation.
///
/// Single left-to-right pass over the tokens (mirroring the
/// [`scan_push_clean_tokens`] value-consumption pattern):
/// - Every token is scanned for mutation tokens first — even when git
///   consumes it as an option value (`git branch --merged -d` must reject).
/// - Required-value options consume the next token unconditionally (even
///   dash-prefixed); a consumed token never triggers list mode
///   (`git branch --sort --list foo` creates `foo`).
/// - Optional-value options consume only their `=value` form; a space value
///   becomes a bare ref name (`git branch --abbrev 10` creates `10`).
/// - `--` and `--end-of-options` end option parsing; remaining tokens are
///   positionals (patterns under list/verify mode, ref names otherwise).
/// - At the end: verify mode, list mode, or no bare positional → read-only;
///   otherwise a bare positional is a ref name → reject.
fn check_git_ref_subcommand(subcommand: &str, sub: &str) -> Result<(), String> {
    let (table, shorts) = match sub {
        "branch" => (GIT_BRANCH_OPTS, GIT_BRANCH_SHORTS),
        _ => (GIT_TAG_OPTS, GIT_TAG_SHORTS),
    };
    // Quote-aware like scan_push_clean_tokens: `--format "%(refname) %(objectname)"`
    // must stay one value token, not split mid-value into a fake ref name.
    let words = split_words_keeping_substitutions(subcommand);
    let mut list = false;
    let mut verify = false;
    let mut saw_name = false;
    let mut after_sep = false;
    let mut consume_next = false;
    for raw in words.iter().skip(1) {
        let w = shell_word(raw);
        if is_ref_mutation_word(&w, table, shorts) {
            return Err(git_mutation_rejection(subcommand));
        }
        if consume_next {
            consume_next = false;
            continue; // git consumes this token as an option value
        }
        if after_sep {
            saw_name = true;
            continue; // post-separator tokens are positionals (still mutation-scanned above)
        }
        if w == "--" || w == "--end-of-options" {
            after_sep = true;
            continue;
        }
        if let Some(kind) = resolve_ref_long_opt(&w, table) {
            match kind {
                RefOpt::Plain | RefOpt::OptVal => {}
                RefOpt::Val => {
                    if !w.contains('=') {
                        consume_next = true;
                    }
                }
                RefOpt::List | RefOpt::ListOptVal => list = true, // ListOptVal is short-only
                RefOpt::ListVal => {
                    list = true;
                    if !w.contains('=') {
                        consume_next = true;
                    }
                }
                RefOpt::Verify => verify = true,
                RefOpt::Mutation => unreachable!("mutation scan rejects mutation tokens first"),
            }
        } else if w.starts_with("--") {
            // Unknown/ambiguous long option — git errors at runtime; fail
            // closed: no list mode, no value consumption, so a following bare
            // word is a ref name.
        } else if w.starts_with('-') && w.len() > 1 {
            let b = w.as_bytes();
            let mut k = 1;
            while k < b.len() {
                let c = b[k] as char;
                match shorts
                    .iter()
                    .find(|(ch, _)| *ch == c)
                    .map(|(_, kind)| *kind)
                {
                    Some(RefOpt::List) => list = true,
                    Some(RefOpt::ListOptVal) => {
                        list = true;
                        if k + 1 < b.len() {
                            break; // rest of token is the attached value
                        }
                    }
                    Some(RefOpt::Verify) => verify = true,
                    Some(RefOpt::Mutation) => {
                        unreachable!("mutation scan rejects mutation tokens first")
                    }
                    _ => {} // Plain or unknown short — fail closed (create-active)
                }
                k += 1;
            }
        } else {
            saw_name = true;
        }
    }
    if verify || list || !saw_name {
        return Ok(());
    }
    Err(format!(
        "⚠️ Read-only mode: `git {subcommand}` is not allowed — it would create a {sub}.\n\
         Suggestion: use `git {sub} --list` or `git {sub} --merged` to list existing ones."
    ))
}

/// For a matched git subcommand, check for mutation verbs across all
/// argument positions (used for `git remote`).
///
/// **Bare-word tokens** (remote verbs like `add`, `remove`, `prune`) are
/// checked only at the first non-flag argument position. They cannot be
/// safely checked in all positions because they can collide with legitimate
/// names — e.g., `git remote show add` is a valid read-only command where
/// `add` is a remote name, not a mutation verb.
///
/// `subcommand` is the pre-extracted subcommand from [`extract_git_subcommand`]
/// (e.g., `"remote -v update"`).
fn check_git_subcommand_mutation(subcommand: &str, mutation_tokens: &[&str]) -> Result<(), String> {
    let words: Vec<&str> = subcommand.split_whitespace().collect();
    // words[0] is the subcommand name (e.g., "remote")

    // Partition mutation tokens into flag-based and bare-word.
    let (flag_tokens, bare_tokens): (Vec<&str>, Vec<&str>) = mutation_tokens
        .iter()
        .copied()
        .partition(|t| t.starts_with('-'));

    let short_chars: Vec<char> = flag_tokens
        .iter()
        .filter_map(|t| {
            let b = t.as_bytes();
            (b.len() == 2 && b[0] == b'-').then(|| b[1] as char)
        })
        .collect();

    // ── Flag-based mutation token check (all positions) ──
    for arg in words.iter().skip(1) {
        let a = shell_word(arg);
        if !a.starts_with('-') {
            continue;
        }
        let is_mutating = if a.starts_with("--") {
            matches_mutation_token(&a, &flag_tokens)
        } else {
            short_chars.iter().any(|c| a[1..].contains(*c))
        };
        if is_mutating {
            return Err(git_mutation_rejection(subcommand));
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
        let is_mutating = matches_mutation_token(first_non_flag_arg, &bare_tokens);
        if is_mutating {
            return Err(git_mutation_rejection(subcommand));
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
    // (e.g., GIT_DIR=/tmp git push, sudo git push, env git push). Basename
    // first, then strip balanced quotes — the same normalization order as
    // dispatch (`first_command_word` → `classify_verb_word`) — so
    // `/usr/bin/git`, `'git'`, and `/usr/bin/'git'` (incl. prefix-forwarded
    // forms like `sudo /usr/bin/'git' push`) all resolve to git and can't
    // bypass the git allowlist.
    let git_idx = super::find_first_command_word_index(&words);
    let Some(git_idx) = git_idx else {
        return String::new();
    };
    let git_word = words[git_idx]
        .rsplit('/')
        .next()
        .expect("rsplit always yields at least one element");
    let git_word = strip_outer_quotes(git_word).map_or(git_word, |(c, _)| c);
    if git_word != "git" {
        return String::new();
    }

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

    // ── Help/version exemption ─────────────────────────────────────
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
                // Dry-run previews changes without modifying Cargo.lock.
                return Ok(());
            }
            return reject(
                trimmed,
                "`cargo update` is not allowed — it modifies Cargo.lock.",
                "use `cargo update --dry-run` to preview what would change without modifying anything; \
                 if the real update is required, state that in your final response — it is outside read-only scope.",
            );
        }
        "generate-lockfile" => {
            return reject(
                trimmed,
                "`cargo generate-lockfile` is not allowed — it creates or overwrites Cargo.lock.",
                "generating a lockfile is outside read-only scope; if it is required, state that in your final response.",
            );
        }
        "run" => {
            return reject(
                trimmed,
                "`cargo run` is not allowed — it may write files.",
                "use `cargo run --help` or `cargo run -h` to inspect the program's CLI without running it; \
                 if running the program is actually required, state that in your final response — it is outside read-only scope.",
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

/// Value of the first of `flags` found in space-separated form, falling back
/// to the `=` form of `equals_prefix` when set. `None` when no flag is
/// present.
fn output_flag_value<'a>(
    parts: &'a [&'a str],
    flags: &[&str],
    equals_prefix: Option<&str>,
) -> Option<&'a str> {
    flags
        .iter()
        .find_map(|f| flag_value(parts, f))
        .or_else(|| equals_prefix.and_then(|p| flag_value_equals(parts, p)))
}

/// Approximate the argv word sed receives after shell parsing: `$'...'`/`$"..."`
/// drop the `$`, then all quotes and backslashes are removed
/// (`\'` → `'`, `\i` → `i`, `'-i'x` → `-ix`). Tokens whose escape semantics
/// cannot be resolved this way are handled by [`is_unprovable_flag_token`].
fn shell_word(word: &str) -> String {
    let word = word
        .strip_prefix('$')
        .filter(|rest| rest.starts_with(['\'', '"']))
        .unwrap_or(word);
    word.replace(['\'', '"', '\\'], "")
}

/// `$'...'`/`$"..."` tokens with backslash escapes are unprovable: escape
/// semantics are shell/version-dependent (`$'\x2di'` → `-i`), so treat them
/// as unprovable flag candidates — mirroring [`classify_verb_word`]'s
/// Unprovable verdict for `$`/`\`-words. Used by the sed in-place gate and
/// the git exec-vector scan (`git log --format=$'%h\t%s'` is over-rejected,
/// an accepted fail-closed contract). `${...}`/`$var` are deliberately not
/// treated as unprovable: they also appear in legit read-only scripts
/// (`sed "s/a/$var/" file`), and their values are not decodable anyway.
fn is_unprovable_flag_token(part: &str) -> bool {
    (part.starts_with("$'") || part.starts_with("$\"")) && part.contains('\\')
}

/// True when any single-dash token in `command` contains a char from
/// `mutation`, combined-cluster aware (`-df`, `-wt`, quoted `'-do'`).
/// `value_taking` chars consume the rest of their token (attached value), so
/// later chars are never misread as flags (`-tw` is `-t w`, `-dio` is
/// `-d -i 'o'`). Long flags and tokens without a leading dash never match.
/// Tokens are normalized as the shell delivers them ([`shell_word`]).
fn has_cluster_char(command: &str, mutation: &[char], value_taking: &[char]) -> bool {
    command.split_whitespace().any(|w| {
        let p = shell_word(w);
        if !p.starts_with('-') || p.starts_with("--") {
            return false;
        }
        let b = p.as_bytes();
        let mut k = 1;
        while k < b.len() {
            let c = b[k] as char;
            if mutation.contains(&c) {
                return true;
            }
            if value_taking.contains(&c) {
                return false; // value-taking flag: rest of token is its value
            }
            k += 1;
        }
        false
    })
}

/// Check if `sed` has an in-place flag in a way that mutates files outside temp.
/// When all file operands after the flag are under temp, returns `false` (allow).
fn has_sed_mutation(command: &str, state: &ValidationState) -> bool {
    let parts: Vec<&str> = command.split_whitespace().collect();
    // In-place flag position: any single-dash flag containing `i`/`I`
    // (-i, -iSUFFIX, -I, -nix/-Ei clusters; no other sed short flag has
    // i/I), or the GNU long form `--in-place[=SUFFIX]` at any unique
    // abbreviation (`--i` is the shortest; no other GNU sed long option
    // starts with `i`). Over-rejection is intentional (fail-closed):
    // attached `-e`/`-f`/`-l` args containing `i` (e.g. `-e's/x/i/'`) and
    // operands literally named like a flag (`sed -- -info.txt`) are
    // rejected too. Tokens are normalized as the shell delivers them
    // ([`shell_word`]), and `$'...'`/`$"..."` tokens with escapes are
    // unprovable flag candidates ([`is_unprovable_flag_token`]).
    let i_pos = parts.iter().position(|part| {
        let p = shell_word(part);
        (p.starts_with('-') && !p.starts_with("--") && p.contains(['i', 'I']))
            || p.starts_with("--i")
            || is_unprovable_flag_token(part)
    });
    let Some(i_pos) = i_pos else {
        return false; // no in-place flag → not a sed mutation
    };

    // Collect file operands after -i.
    // Skip flags (-n, -e, etc.), sed expressions, and backup-extension tokens.
    // Sed expressions and backup extensions are non-absolute tokens before the
    // first absolute-path file operand. Once we've seen a file operand, all
    // subsequent non-flag tokens are treated as file operands (multi-file).
    // The absolute check normalizes via [`shell_word`], so quoted/escaped
    // paths (e.g. '/etc/passwd') count as operands — which also classifies
    // `/`-addressed scripts (e.g. '/pat/d') as operands, an accepted fail-closed
    // over-rejection matching the unquoted behavior. Unprovable `$'...'`/`$"..."`
    // tokens with escapes ([`is_unprovable_flag_token`]) also count as
    // operands — they may resolve to a workspace path.
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
        if Path::new(&shell_word(part)).is_absolute() || is_unprovable_flag_token(part) {
            seen_file_operand = true;
            file_operands.push(part);
        }
        // Non-absolute tokens are sed expressions or backup extensions — skip
    }

    if file_operands.is_empty() {
        return true; // in-place flag without file operand → reject (conservative)
    }

    file_operands.iter().any(|p| writes_outside_temp(p, state))
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
        return writes_outside_temp(of_val, state);
    }
    false
}

/// Curl short flags that take an attached or separate value. Their attached
/// values consume the rest of the token, so `-do` (data "o") and
/// `-HContent-Type:o` are never misread as `-o`, and `-dO` is never misread
/// as `-O`. `-h` (help) takes an optional attached subject (`-ho` is help for
/// "o", not `-o`). `-o` itself is handled by [`has_output_mutation`], not
/// listed here.
const CURL_VALUE_TAKING: &[char] = &[
    'A', 'b', 'c', 'C', 'd', 'D', 'e', 'E', 'F', 'H', 'h', 'K', 'm', 'P', 'Q', 'r', 't', 'T', 'u',
    'U', 'w', 'x', 'X', 'y', 'Y', 'z',
];

/// Output-flag family for one tool: single-dash chars that unconditionally
/// write to CWD (`-O`), the temp-gated `-o` output char, value-taking chars,
/// and long flags that unconditionally write to CWD.
struct OutputFlagSpec {
    always_blocked_short: &'static [char],
    output_short: char,
    value_taking: &'static [char],
    always_blocked_long: &'static [&'static str],
}

const CURL_OUTPUT_FLAGS: OutputFlagSpec = OutputFlagSpec {
    always_blocked_short: &['O'],
    output_short: 'o',
    value_taking: CURL_VALUE_TAKING,
    always_blocked_long: &["--remote-name", "--remote-name-all"],
};

const BASE64_OUTPUT_FLAGS: OutputFlagSpec = OutputFlagSpec {
    always_blocked_short: &[],
    output_short: 'o',
    value_taking: &['i', 'b'],
    always_blocked_long: &[],
};

/// Check if a curl/base64-style command has output flags that write outside
/// temp. Every output flag is evaluated — no early return on a temp-gated one:
/// curl takes the first `-o`/`--output` and adds a CWD write per URL with `-O`,
/// while base64 takes the last `-o`/`--output`, so an outside-temp write
/// anywhere is fail-closed regardless of which flag ultimately wins.
/// `-O`/`--remote-name`/`--remote-name-all` always write to CWD; `-o`/`--output`
/// (space and `=` forms) are temp-gated with attached or next-token values;
/// value-taking chars consume the rest of their token (`-do` is data "o",
/// `-dio` is `-d -i 'o'`). Long flags and `=`-forms match after shell
/// normalization ([`shell_word`]), so quoted spellings (`'-so'`, `--'output'`)
/// are seen; `$'...'`/`$"..."` tokens with escapes are unprovable flag
/// candidates — same fail-closed contract as the sed gate.
fn has_output_mutation(command: &str, state: &ValidationState, spec: &OutputFlagSpec) -> bool {
    let parts: Vec<&str> = command.split_whitespace().collect();

    for (i, part) in parts.iter().enumerate() {
        let w = shell_word(part);
        if spec.always_blocked_long.contains(&w.as_str()) {
            return true;
        }
        if let Some(path) = w.strip_prefix("--output=") {
            if writes_outside_temp(path, state) {
                return true;
            }
        } else if w == "--output"
            && let Some(next) = parts.get(i + 1)
            && writes_outside_temp(next, state)
        {
            return true;
        }
    }
    // ANSI-C `$'...'`/`$"..."` tokens with escapes are unprovable flag
    // candidates (`$'\x2do'` delivers `-o`) — same fail-closed contract as
    // the sed gate.
    if parts.iter().any(|p| is_unprovable_flag_token(p)) {
        return true;
    }

    // Single-dash forms, standalone or in a combined cluster (`-so`, `-sO`,
    // `-OJ`, `-LO`, `-so/tmp/x`, quoted `'-so'`).
    for (i, part) in parts.iter().enumerate() {
        let p = shell_word(part);
        if !p.starts_with('-') || p.starts_with("--") || p.len() < 2 {
            continue;
        }
        let b = p.as_bytes();
        let mut k = 1;
        while k < b.len() {
            let c = b[k] as char;
            if spec.always_blocked_short.contains(&c) {
                return true;
            }
            if c == spec.output_short {
                let path = if k + 1 < b.len() {
                    &p[k + 1..]
                } else {
                    parts.get(i + 1).copied().unwrap_or("")
                };
                if !path.is_empty() && writes_outside_temp(path, state) {
                    return true;
                }
                break; // -o value consumed (attached or next token)
            }
            if spec.value_taking.contains(&c) {
                break; // value-taking flag: rest of token is its value
            }
            k += 1;
        }
    }

    false
}

/// Check if curl has output flags that write outside temp (see
/// [`has_output_mutation`]). `-o <path>`/`--output <path>` is allowed when the
/// path is under temp; `-O`/`--remote-name`/`--remote-name-all` always write
/// to CWD and are blocked, standalone or in a cluster (`-sO`, `-OJ`, `-LO`).
fn has_curl_mutation(command: &str, state: &ValidationState) -> bool {
    has_output_mutation(command, state, &CURL_OUTPUT_FLAGS)
}

/// Check if `wget` output flags write outside temp.
/// `-O <path>` / `--output-document <path>` / `-P <path>` / `--directory-prefix <path>`
/// are allowed when the path is under temp.
/// Without output flags, wget writes to CWD → always blocked.
fn has_wget_mutation(command: &str, state: &ValidationState) -> bool {
    let parts: Vec<&str> = command.split_whitespace().collect();

    // -O/--output-document and -P/--directory-prefix (space and = forms):
    // allowed when the path is under temp.
    if let Some(path) = output_flag_value(
        &parts,
        &["-O", "--output-document"],
        Some("--output-document="),
    ) {
        return writes_outside_temp(path, state);
    }
    if let Some(path) = output_flag_value(
        &parts,
        &["-P", "--directory-prefix"],
        Some("--directory-prefix="),
    ) {
        return writes_outside_temp(path, state);
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

/// Check if `base64` writes output outside temp (see [`has_output_mutation`]).
/// The decode gate is deliberately absent: `-o` writes a file in encode mode
/// too. `-i`/`-b` are value-taking on BSD/macOS (`-dio` is `-d -i 'o'`, not an
/// output flag); GNU base64 has no `-o`/`-i` at all, so no write happens there.
fn has_base64_mutation(command: &str, state: &ValidationState) -> bool {
    has_output_mutation(command, state, &BASE64_OUTPUT_FLAGS)
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
            // Match the session shell env (TMPDIR = super::TMPDIR_BASELINE); TMP/TEMP unset.
            temp_vars: vec![(
                "TMPDIR".to_string(),
                super::super::TMPDIR_BASELINE.to_string(),
            )],
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
            // branch mutation flags as first arg (bypasses name-creation check)
            ("git branch --track feature", false),
            ("git branch --no-track feature", false),
            ("git branch -t", false),
            ("git branch --unset-upstream", false),
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
            // fsck: --lost-found writes dangling objects into .git/lost-found/
            // (git abbreviates long options, so --lost/--l trigger it too;
            // quotes are stripped by shell_word);
            // --no-lost-found and the read-only flags stay allowed
            ("git fsck", true),
            ("git fsck --strict --full", true),
            ("git fsck --no-lost-found", true),
            ("git fsck --lost-found", false),
            ("git fsck \"--lost-found\"", false),
            ("git fsck --lost", false),
            ("git fsck --l", false),
            // --output file-write vector (diff/log family writes the file;
            // blame-class truncates the target to 0 bytes; dash values are
            // valid filenames)
            ("git diff --output=/tmp/x", false),
            ("git show --output /tmp/x", false),
            ("git log -p --output=/tmp/x", false),
            ("git blame --output=src/lib.rs", false),
            ("git diff-tree --output=/tmp/x", false),
            ("git diff --output -1", false),
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

    /// Combined single-dash flag clusters: mutation chars inside a cluster
    /// (`git branch -df`, `-mv`, `git tag -am`, `git hash-object -wt blob`,
    /// `git config -e`) are detected, while read-only clusters (`-vv`, `-a`,
    /// `-n1`) and quoted spellings stay correctly classified.
    #[test]
    fn git_combined_short_flag_clusters() {
        let cases = [
            // branch: delete/force clusters
            ("git branch -df feature", false),
            ("git branch -fd feature", false),
            ("git branch -Dd feature", false),
            ("git branch '-df' feature", false),
            ("git branch -v '-df' feature", false), // quoted cluster after a safe flag
            // branch: move/copy/force clusters
            ("git branch -mv old new", false),
            ("git branch -cv old new", false),
            ("git branch -fv main origin/main", false),
            // branch: attached -u upstream value
            ("git branch -uorigin/main", false),
            // tag: annotate/force/message clusters
            ("git tag -am 'msg' v1.0", false),
            ("git tag -afm 'msg' v1.0", false),
            ("git tag -fam 'msg' v1.0", false),
            ("git tag -ma v1.0", false),
            ("git tag -mprobe-msg v1.0", false),
            ("git tag -F/file v1.0", false),
            ("git tag '-am' 'msg' v1.0", false),
            ("git tag -n '-am' 'msg' v1.0", false), // quoted cluster after a safe flag
            // hash-object: -w inside a cluster with the -t type flag
            ("git hash-object -wt blob a.txt", false),
            ("git hash-object -wtblob a.txt", false),
            ("git hash-object '-wt' blob a.txt", false),
            ("git hash-object -w -t blob a.txt", false),
            // config: -e edit short form (single and quoted)
            ("git config -e", false),
            ("git config '-e'", false),
            ("git config -f /tmp/cfg -e", false),
            // config: quoted long write forms (shell-stripped quotes deliver
            // --edit/--unset); quoted read forms stay allowed
            ("git config --'edit'", false),
            ("git config --'unset' key", false),
            ("git config --'list'", true),
            // read-only clusters stay allowed
            ("git branch -v", true),
            ("git branch -vv", true),
            ("git branch -a", true),
            ("git branch -r", true),
            ("git tag -n", true),
            ("git tag -n1", true),
            ("git hash-object -t blob a.txt", true),
            ("git hash-object --stdin", true),
            ("git config -l", true),
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

    /// Tests that all git branch mutation options (long + short, from the
    /// git 2.50.1 table) are rejected via [`check_git_ref_subcommand`],
    /// including their `=value` forms.
    #[test]
    fn git_branch_mutation_flags_rejected() {
        for &(flag, kind) in GIT_BRANCH_OPTS {
            if kind == RefOpt::Mutation {
                assert_rejected(&format!("git branch {flag} feature"));
                assert_rejected(&format!("git branch {flag}=x feature"));
            }
        }
        for &(flag, kind) in GIT_BRANCH_SHORTS {
            if kind == RefOpt::Mutation {
                assert_rejected(&format!("git branch -{flag} feature"));
            }
        }
    }

    /// Tests that all git tag mutation options (long + short, from the
    /// git 2.50.1 table) are rejected via [`check_git_ref_subcommand`],
    /// including their `=value` forms.
    #[test]
    fn git_tag_mutation_flags_rejected() {
        for &(flag, kind) in GIT_TAG_OPTS {
            if kind == RefOpt::Mutation {
                assert_rejected(&format!("git tag {flag} v1.0"));
                assert_rejected(&format!("git tag {flag}=x v1.0"));
            }
        }
        for &(flag, kind) in GIT_TAG_SHORTS {
            if kind == RefOpt::Mutation {
                assert_rejected(&format!("git tag -{flag} v1.0"));
            }
        }
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
            // branch: value-consuming space forms (value consumed, no bare name)
            ("git branch --format %(refname)", true),
            ("git branch --sort committerdate", true),
            // branch/tag: quoted multi-word values stay one consumed token
            ("git branch --format \"%(refname) %(objectname)\"", true),
            ("git branch --format=\"%(refname) %(objectname)\"", true),
            ("git tag --format \"%(refname) %(objectname)\"", true),
            ("git tag --format=\"%(refname) %(objectname)\"", true),
            // branch: optional-value `=`-forms (read-only without a name)
            ("git branch --abbrev=10", true),
            ("git branch --color=always", true),
            ("git branch --column=dense", true),
            // branch: unambiguous long-option abbreviations of list filters
            ("git branch --mer main", true),
            ("git branch --con HEAD", true),
            ("git branch --no-contains HEAD", true),
            ("git branch --no-merged main", true),
            // branch: --list after a bare word turns it into a pattern (loosening)
            ("git branch feature --list", true),
            // branch: listing-only actions without a name
            ("git branch --show-current", true),
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
            ("git tag -v v1.0", true), // verify mode: positionals are verify targets
            ("git tag --verify v1.0", true),
            ("git tag -n 1", true), // -n optional value NOT consumed → pattern
            ("git tag feature --list", true), // list after bare word → pattern
        ];

        run_cases(&cases);
    }

    /// Tests that every branch/tag name-creation and mutation vector from the
    /// git 2.50.1 tables is rejected: optional-value space forms, separators,
    /// unambiguous long abbreviations, consumed-token ordering, and negated
    /// flags that leave a bare ref name.
    #[test]
    fn git_branch_tag_creation_bypass_rejected() {
        let cases = [
            // optional-value space forms leave a bare ref name → create
            ("git branch --abbrev 10", false),
            ("git branch --color always", false),
            ("git branch --column dense", false),
            // separators end option parsing — a bare name still creates
            ("git branch -- foo", false),
            ("git branch --end-of-options foo", false),
            ("git tag --end-of-options foo", false),
            // negated flags are create-active (no value, no list mode)
            ("git branch --no-list foo", false),
            ("git branch --no-verbose foo", false),
            ("git branch --no-delete foo", false),
            ("git branch --no-move foo", false),
            ("git branch --no-force foo", false),
            ("git branch --no-points-at foo", false),
            ("git tag --no-annotate foo", false),
            ("git tag --no-force foo", false),
            // unambiguous long-option abbreviations of mutation flags
            ("git branch --forc foo", false),
            ("git branch --tr foo", false),
            ("git branch --uns foo", false),
            ("git branch --mov foo", false),
            ("git branch --del foo", false),
            ("git branch --edit-d foo", false),
            ("git tag --mes foo", false),
            // unknown long/short flags fail closed → bare word is a name
            ("git branch --mes foo", false),
            ("git branch --bogus foo", false),
            // consumed tokens never trigger list mode
            ("git branch --sort --merged foo", false),
            ("git branch --sort --list foo", false),
            ("git branch --format %(refname) foo", false),
            ("git tag --sort --list foo", false),
            // quoted value consumed; a trailing bare word is still a name
            (
                "git branch --format \"%(refname) %(objectname)\" foo",
                false,
            ),
            ("git tag --format \"%(refname) %(objectname)\" foo", false),
            // mutation tokens still reject when git consumes them as values
            ("git branch --merged -d", false),
            ("git branch --sort -d foo", false),
            // `-a`/`-r` are list-only: a bare name after them is rejected
            ("git branch -a foo", false),
            ("git branch -r foo", false),
            // tag: create-active flags without list mode
            ("git tag -i foo", false),
            ("git tag --column foo", false),
            ("git tag --sort=key foo", false),
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
            // curl combined short-flag clusters
            ("curl -so out URL", false),
            ("curl -sO URL", false),
            ("curl -OJ URL", false),
            ("curl -LO URL", false),
            ("curl '-so' out URL", false),
            ("curl -so/tmp/x URL", true),
            ("curl -do out URL", true), // -d consumes 'o' (data), not output
            ("curl -HContent-Type:o URL", true), // -H value contains 'o'
            ("curl -ho URL", true),     // -h help subject 'o', not output
            ("curl $'\\x2do' /etc/passwd URL", false), // unprovable → -o delivered
            // curl: every output flag evaluated — -O/--remote-name* write to
            // CWD even after a temp-gated -o/--output (no early return), and a
            // later outside-temp flag rejects even when curl would first-wins
            // (fail-closed over-rejection)
            ("curl -O --output /tmp/x URL", false),
            ("curl -o out --output /tmp/x URL", false),
            ("curl -o /tmp/x --output out URL", false),
            ("curl -o /tmp/x --output /tmp/y URL", true),
            ("curl --output=/tmp/x URL", true),
            ("curl --output=out URL", false),
            ("curl --'output' out URL", false), // quoted long form
            ("curl --remote-name-all URL", false), // per-URL CWD writes
            // tar
            ("tar -tf archive.tar.gz", true),
            ("tar -xzf archive.tar.gz", false),
            ("tar -czf archive.tar.gz dir/", false),
            ("tar --list -f archive.tar.gz", true),
            // base64
            ("base64 -d file.txt", true),
            ("base64 -d -o out.bin file.txt", false),
            ("base64 --decode --output out.bin file.txt", false),
            // base64 combined short-flag clusters
            ("base64 -do out", false),
            ("base64 '-do' out", false),
            ("base64 -do/tmp/x", true),
            ("base64 -o out file", false), // encode-mode output write
            ("base64 -dio out", true),     // -i consumes 'o' (input file) on BSD
            ("base64 $'\\x2do' /etc/passwd", false), // unprovable → -o delivered
            // base64: last output flag wins — any outside-temp write rejects
            // (also covers the --output= form)
            ("base64 --output /tmp/x -o out file", false),
            ("base64 -o out --output /tmp/x file", false), // fail-closed over-rejection
            ("base64 -o /tmp/x --output /tmp/y file", true),
            ("base64 --output=/tmp/x file", true),
            ("base64 --output=out file", false),
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
            ("12>", TokenKind::Redirect { needs_target: true }),
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
            // {digit}{op}{digit} shapes: fd + target merged, no separate word
            ("1>0", NO_TARGET),
            ("2>1", NO_TARGET),
            ("2>0", NO_TARGET),
            ("3<4", NO_TARGET),
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
            ("GIT_DIR=/tmp git status", false), // GIT_* env rejected (exec vector)
        ];

        run_cases(&cases);
    }

    /// Git exec vectors: flags/config/env channels that run programs are
    /// rejected, even on allowlisted subcommands. Includes the analyst-flagged
    /// gaps (repo redirect, --config-env/--config-file, export channel,
    /// env/sudo-wrapped forms) and documented fail-closed over-rejections.
    #[test]
    fn git_exec_vectors_rejected() {
        let cases = [
            // ── GIT_* env bindings (current-shell, export, and wrapped) ──
            ("GIT_DIR=/tmp git status", false),
            ("GIT_EXTERNAL_DIFF=/bin/echo git diff", false),
            ("GIT_SSH_COMMAND=/bin/echo git ls-remote origin", false),
            ("GIT_CONFIG_COUNT=1 git log", false),
            ("GIT_EXEC_PATH=/tmp/evil git request-pull", false),
            ("GIT_ASKPASS=/bin/echo git ls-remote origin", false),
            ("GIT_TRACE=/tmp/trace git status", false),
            ("export GIT_EXTERNAL_DIFF=/bin/echo && git diff", false),
            (
                "export GIT_SSH_COMMAND=/bin/echo; git ls-remote origin",
                false,
            ),
            ("env GIT_EXTERNAL_DIFF=/bin/echo git diff", false),
            ("sudo GIT_DIR=/tmp git status", false),
            ("GIT_PAGER=/bin/echo git log", true), // carve-out: no TTY → pager never spawns
            // ── -c/-C global options (position-sensitive: log/diff -c stay allowed) ──
            ("git -c diff.external=/bin/echo diff", false),
            ("git -c core.fsmonitor=/bin/echo status", false),
            ("git -c user.name=me log", false),
            ("git -cfoo=bar status", false), // attached form
            ("git -pc status", false),       // cluster
            ("git -C /tmp/repo status", false),
            ("git -C/tmp/repo diff", false),   // attached form
            ("git -pC /tmp/repo diff", false), // cluster
            // ── repo-redirect / config-injection long globals ──
            ("git --git-dir=/tmp/evil/.git diff", false),
            ("git --git-dir /tmp/evil status", false),
            ("git --work-tree=/tmp/evil status", false),
            ("git --config-env=core.fsmonitor=F status", false),
            ("git --config-file=/tmp/evil status", false),
            ("git --exec-path=/tmp/evil request-pull", false),
            // ── exec flags (anywhere in the command; --no-* forms stay allowed) ──
            ("git diff --ext-diff", false),
            ("git diff --no-index --ext-diff /tmp/a /tmp/b", false),
            ("git log -p --textconv", false),
            ("git log --show-signature", false),
            ("git cat-file --filters HEAD:src/lib.rs", false),
            ("git cat-file --text HEAD:a.txt", false), // abbrev of --textconv: execs the driver
            ("git ls-remote --upload-pack=/bin/echo origin", false),
            ("git ls-remote --upload-pack /bin/echo origin", false),
            ("git push --dry-run --exec=/bin/echo origin", false), // extension bypass
            ("git stash show --textconv", false),                  // extension bypass
            (
                "git push --dry-run --receive-pack=/bin/echo origin main",
                false,
            ),
            (
                "git push --dry-run --receive-pack /bin/echo origin main",
                false,
            ),
            // ── git abbreviated long-option prefixes (--upload-p= == --upload-pack=) ──
            ("git ls-remote --upload-p=/bin/echo origin", false),
            ("git ls-remote --upload-pa /bin/echo origin", false),
            ("git ls-remote --exe=/bin/echo origin", false),
            ("git push --dry-run --exe=/bin/echo origin", false),
            ("git push --dry-run --exe /bin/echo origin", false),
            (
                "git push --dry-run --receive-p=/bin/echo origin main",
                false,
            ),
            ("git grep --open-files-in-p=/bin/echo pattern", false),
            ("git grep --open-files-in-p /bin/echo pattern", false),
            // defense-in-depth (this git build rejects these spellings itself)
            ("git diff --ext-d", false),
            ("git log --textc", false),
            ("git log --show-signat", false),
            ("git cat-file --filt HEAD:src/lib.rs", false), // ambiguous in git, blocked anyway
            // ── grep -O / --open-files-in-pager ──
            ("git grep -O /bin/echo pattern", false),
            ("git grep -O/bin/echo pattern", false),
            ("git grep -nO /bin/echo pattern", false),
            ("git grep -IO /bin/echo pattern", false),
            ("git grep --open-files-in-pager=/bin/echo pattern", false),
            ("git grep --open-files-in-pager pattern", false),
            // ── help viewers (web/info/man all exec external programs) ──
            ("git help --web", false),
            ("git help -w", false),
            ("git help -aw", false),
            ("git help -m git", false),
            ("git help -i git", false),
            ("git help --man git", false),
            ("git help --info git", false),
            ("git help --m git", false),
            // ── full-path git (basename allowlist bypass) ──
            ("/usr/bin/git push", false),
            ("/usr/bin/git -C /tmp status", false),
            // ── documented fail-closed over-rejections ──
            ("git grep -e '--ext-diff' pattern", false), // literal pattern
            ("git grep -e '-Ofoo' pattern", false),      // literal pattern
            ("git log --format=$'%h\\t%s'", false),      // ANSI-C unprovable token
            ("git log -- --ext-diff", false),            // pathname named like a flag
        ];

        run_cases(&cases);
    }

    /// Positive controls for the exec-vector scan: ordinary read-only git and
    /// the legitimate spellings that must NOT be caught (position/subcommand
    /// scoping, the bare `--` path separator, the benign `--text` flag,
    /// --no-* disabling forms, GIT_PAGER carve-out).
    #[test]
    fn git_exec_vector_positives() {
        let cases = [
            // Ordinary read-only git still passes
            ("git status", true),
            ("git log", true),
            ("git diff", true),
            ("git show HEAD", true),
            ("git blame src/lib.rs", true),
            // bare `--` path separator is never an option
            ("git log -- src/lib.rs", true),
            ("git log --oneline -- path", true),
            ("git diff -- file", true),
            ("git show HEAD -- file", true),
            ("git grep -- pattern", true),
            // exact `--text` is the benign -a flag where git has it
            ("git grep --text pattern", true),
            ("git diff --text", true),
            ("git log --text", true),
            ("git blame --text src/lib.rs", true),
            // Position-sensitive: -c after the subcommand is combined diff
            ("git log -c", true),
            ("git diff -c", true),
            ("git log --oneline -c", true),
            // -w on diff/log is ignore-whitespace (help-scoped -w only)
            ("git diff -w", true),
            ("git log -w", true),
            // -O on log is a benign order-file read (grep-scoped -O only)
            ("git log -Oorderfile", true),
            // grep short flags without uppercase -O
            ("git grep -o pattern", true),
            ("git grep -n pattern", true),
            ("git grep -I pattern", true),
            ("git grep -c pattern", true),
            ("git grep -C3 pattern", true),
            ("git grep --regexp=-Ofoo pattern", true), // long form spared
            // --no-* disabling forms stay allowed
            ("git diff --no-ext-diff", true),
            ("git diff --no-textconv", true),
            ("git log --no-show-signature", true),
            // --filter exact stays allowed (legit flag shadowing --filters)
            ("git log --filter=blob:none", true),
            ("git rev-list --filter=blob:none HEAD", true),
            ("git cat-file --filter blob:none HEAD", true),
            // help non-viewer flags and topics
            ("git help", true),
            ("git help -a", true),
            ("git help --all", true),
            ("git help -g", true),
            ("git help -c core.pager", true),
            ("git help --config", true),
            // regular ls-remote flags stay allowed
            ("git ls-remote --heads origin", true),
            // paginate globals (no c/C)
            ("git -p log", true),
            ("git -P log", true),
            // GIT_PAGER carve-out (pager never spawns on piped stdout)
            ("GIT_PAGER=less git log", true),
            // quoted args that only look like flags (no backslash escapes)
            ("git log --format=$'%h'", true),
            ("git log --grep=--ext-diff", true),
        ];

        run_cases(&cases);
    }

    /// Path-spelled git invocations route to the git layer exactly like plain
    /// `git`: basename dispatch + basename-matched subcommand extraction, so
    /// mutating forms reject and read-only forms stay allowed. Per-spelling
    /// extraction output is pinned by `test_extract_git_subcommand`; this
    /// matrix pins the end-to-end routing per spelling shape.
    #[test]
    fn git_path_spelled_invocations() {
        let cases = [
            // absolute / relative paths
            ("/usr/bin/git init", false),
            ("/usr/bin/git status", true),
            ("./git init", false),
            ("./git log", true),
            // quoted git and quotes inside the final path component are
            // rejected fail-closed: bare forms at the verb gate, prefix-
            // forwarded forms via git-layer validation
            ("'git' init", false),
            ("/usr/bin/'git' init", false),
            ("sudo /usr/bin/'git' reset --hard", false),
            ("env /usr/bin/'git' commit -m x", false),
            ("sudo /usr/bin/'git' status", true),
            // prefixes in front of the path
            ("sudo /usr/bin/git init", false),
            ("sudo /usr/bin/git status", true),
            ("env /usr/bin/git push", false),
            // combined short-flag cluster through the same gate
            ("/usr/bin/git branch -df feature", false),
        ];

        run_cases(&cases);
    }

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
                name: "absolute path",
                input: "/usr/bin/git status",
                expected: "status",
            },
            Case {
                name: "absolute path with global flag",
                input: "/opt/homebrew/bin/git -C /repo diff",
                expected: "diff",
            },
            Case {
                name: "prefixed absolute path",
                input: "sudo /usr/bin/git push",
                expected: "push",
            },
            Case {
                name: "relative path",
                input: "./git log",
                expected: "log",
            },
            Case {
                name: "quoted git",
                input: "'git' branch -d feature",
                expected: "branch -d feature",
            },
            Case {
                name: "quotes in final path component",
                input: "/usr/bin/'git' status",
                expected: "status",
            },
            Case {
                name: "prefixed quotes in final path component",
                input: "sudo /usr/bin/'git' push",
                expected: "push",
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
            // copying into the workspace stays blocked).
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
            // sed in-place family: attached suffix, clusters, uppercase -I
            ("sed -ix 's/a/b/' /tmp/file", true),
            ("sed -ix 's/a/b/' /etc/passwd", false),
            ("sed -nix 's/a/b/' /etc/passwd", false),
            ("sed -I 's/a/b/' /tmp/file", true),
            ("sed -I 's/a/b/' /etc/passwd", false),
            ("sed -nIx 's/a/b/' /etc/passwd", false),
            // GNU long form --in-place (token-level; behavioral check needs GNU sed)
            ("sed --in-place 's/a/b/' /tmp/file", true),
            ("sed --in-place=bak 's/a/b/' /tmp/file", true),
            ("sed --in-place 's/a/b/' /etc/passwd", false),
            ("sed --in-place=bak 's/a/b/' /etc/passwd", false),
            // GNU getopt_long abbreviations (--i is the shortest unique prefix)
            ("sed --i 's/a/b/' /etc/passwd", false),
            // Quote-wrapped / concatenated flags (shell strips quotes)
            ("sed '-i' 's/a/b/' /tmp/file", true),
            ("sed '-i' 's/a/b/' /etc/passwd", false),
            ("sed '-i'x 's/a/b/' /tmp/file", true), // concatenates to -ix
            ("sed '-i'x 's/a/b/' /etc/passwd", false),
            ("sed '-nix' 's/a/b/' /etc/passwd", false),
            ("sed --'in-place' 's/a/b/' /tmp/file", true), // → --in-place
            ("sed --'in-place' 's/a/b/' /etc/passwd", false),
            // Backslash-escaped and ANSI-C-quoted flags (shell delivers -i/-ix)
            ("sed \\-ix 's/a/b/' /tmp/file", true),
            ("sed \\-ix 's/a/b/' /etc/passwd", false),
            ("sed \\--in-place 's/a/b/' /etc/passwd", false),
            ("sed $'-ix' 's/a/b/' /etc/passwd", false),
            // $'...'/$"..." tokens with escapes are unprovable flag candidates
            ("sed $'\\x2di' 's/a/b/' /tmp/file", true), // → -i; temp still allowed
            ("sed $'\\x2di' 's/a/b/' /etc/passwd", false),
            ("sed $'\\055i' 's/a/b/' /etc/passwd", false), // octal escape
            ("sed $'\\x2dix' 's/a/b/' /etc/passwd", false), // → -ix
            ("sed $'\\u002di' 's/a/b/' /etc/passwd", false), // any escape → unprovable
            ("sed $'\\\\x2di' 's/a/b/' /etc/passwd", false), // literal backslash → over-rejected
            ("sed $\"\\x2di\" 's/a/b/' /etc/passwd", false), // locale quoting → over-rejected
            // $'...' operands with escapes count as operands (may be a path)
            ("sed -i 's/a/b/' $'\\x2fetc\\x2fpasswd' /tmp/file", false),
            // Documented over-rejections (fail-closed contract)
            ("sed -e's/x/i/' file", false), // attached -e script arg
            ("sed -- -info.txt", false),    // operand literally named like a -i flag
            // Quote-wrapped operands (normalized for the absolute check)
            ("sed -i 's/a/b/' '/tmp/file'", true),
            ("sed -i 's/a/b/' '/etc/passwd' /tmp/file", false), // mixed
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
            // base64 -o to temp → allowed (encode or decode mode)
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

    // ── Trailing-backslash token (regression: panic in expand_vars) ──
    // A word-final lone `\` must fail closed (reject), never panic dispatch.
    #[test]
    fn trailing_backslash_token_fails_closed() {
        for cmd in [
            // Exact dispatch trigger: split_whitespace() turns `./-\ 1` into
            // the word `./-\`, whose trailing backslash cannot resolve.
            "rm -f /tmp/outtest/-1 ./-\\ 1 2>/dev/null",
            "rm /tmp/scratch-\\",
            "touch /tmp/ok \\",
        ] {
            assert_rejected(cmd);
        }
    }

    /// Tests that ALL entries in [`TEMP_MUTATORS`] are allowed with temp paths
    /// and rejected with non-temp paths, preventing coverage drift.
    #[test]
    fn all_temp_mutators_allowed_with_temp_paths() {
        for &cmd in TEMP_MUTATORS {
            let args = match cmd {
                "cp" | "mv" => "/tmp/a /tmp/b",
                "zip" => "/tmp/out.zip /tmp/file1",
                _ => "/tmp/scratch.txt",
            };
            ok(&format!("{cmd} {args}"));
        }
    }

    /// Tests that ALL entries in [`TEMP_MUTATORS`] are rejected when used
    /// with a non-temp absolute path, preventing accidental test drift.
    #[test]
    fn all_temp_mutators_rejected_with_non_temp() {
        for &cmd in TEMP_MUTATORS {
            assert_rejected(&format!("{cmd} /etc/blocked_test"));
        }
    }

    /// Renders every [`MUTATOR_CHECKS`] rejection template so malformed
    /// `{verb}` placeholders fail in CI instead of at runtime, and sanity-
    /// checks the static git always-mutate rows.
    #[test]
    fn mutator_table_rows_render() {
        for check in MUTATOR_CHECKS {
            for &verb in check.verbs {
                let rendered = check.rejection.replace("{verb}", verb);
                assert!(
                    !rendered.contains('{'),
                    "unsubstituted placeholder in rejection: {rendered}"
                );
                assert!(
                    rendered.contains(verb),
                    "verb missing from rendered rejection: {rendered}"
                );
            }
        }
        for &(_, why, suggestion) in GIT_ALWAYS_MUTATE {
            assert!(!why.is_empty() && !suggestion.is_empty());
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
            // Dangling `<<` markers (no delimiter): bash errors on these and
            // executes nothing, so they are non-mutating — must not panic.
            ("cat <<", true),
            ("cat << ", true),
            ("cat <<-", true),
            ("cat <<- ", true),
            ("3<< ", true),
            ("3<<-", true),
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
            // substitution.
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
            // "verifiably closed").
            ("export TMPDIR=/etc\necho $(touch $TMPDIR/x)", false),
            ("export TMPDIR=/etc\ncat $(echo hi > $TMPDIR/out)", false),
            // Same via the workspace-root spelling.
            (
                "export TMPDIR=/__mahbot_readonly_test_ws__\necho $(echo hi > $TMPDIR/out)",
                false,
            ),
            // Env-prefix binding in a prior segment poisons a later
            // substitution (env-prefix form).
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
            // Allowed read forms (phase 3).
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
            // --output-indicator-* share the --output prefix but do not write
            ("git diff --output-indicator-new=+", true),
            ("git diff --output-indicator-old=-", true),
            (
                "git diff --output-indicator-new=+ --output-indicator-old=-",
                true,
            ),
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
            // --output writes a file even on read forms (stash show/list)
            ("git stash show --output=/tmp/x", false),
            ("git stash list --output=/tmp/x", false),
            ("git submodule foreach", false),
            ("git submodule update", false),
            ("git submodule add https://example.com/repo.git", false),
        ];

        run_cases(&cases);
    }

    /// Value-aware dry-run/force detection for `git push`/`git clean`: a
    /// value-taking `-o`/`--push-option` (push) or `-e`/`--exclude` (clean)
    /// consumes dry-run/force-looking values, so a real mutation can never
    /// masquerade as a preview.
    #[test]
    fn git_push_clean_value_aware_pins() {
        let cases = [
            // short attached value
            ("git push -onope origin main", false),
            ("git push -o=n origin main", false),
            ("git clean -e.nope", false),
            ("git clean -en", false),
            ("git clean -e=n", false),
            // short separated value (next token consumed even if dash-leading)
            ("git push -o -nope origin main", false),
            ("git push -o --dry-run origin main", false),
            ("git push -o -nu origin main", false),
            ("git push -o -fn origin main", false),
            ("git push -o -n -f origin main", false),
            ("git clean -e -n", false),
            ("git clean -e --dry-run", false),
            // long separated value + unambiguous abbreviations
            ("git push --push-option --dry-run origin main", false),
            ("git push --push-opti --dry-run origin main", false),
            ("git clean --exclude -n", false),
            ("git clean --excl -n", false),
            // `--repo` value-taker (positional later wins; verified real push)
            ("git push --repo --dry-run origin main", false),
            ("git push --rep --dry-run origin main", false),
            ("git push --repo=--dry-run origin main", false),
            // ambiguous long prefix fails closed (value-taking)
            ("git push --p --dry-run origin main", false),
            // quoted forms
            ("git push -o \"x -n\" origin main", false),
            ("git push -o \"x --dry-run\" origin main", false),
            ("git push -o \"--dry-run\" origin main", false),
            ("git push '-o' --dry-run origin main", false),
            ("git clean '-e' -n", false),
            // `--` pathspec / separator
            ("git clean -- -n", false),
            ("git clean '--' -n", false),
            // pre-satisfied lock-in
            ("git push -o nope origin main", false),
            // documented fail-closed over-rejection: quoted genuine dry-run
            ("git push '--dry-run' origin main", false),
            // `git clean --e -n` is covered by the exec-flag layer (`--e` is a
            // prefix of `--ext-diff`), shadowing the scanner's abbrev rule.
            ("git clean --e -n", false),
            // genuine dry-runs stay allowed (value consumed, no force)
            ("git push --dry-run -onope origin main", true),
            ("git push -o nope --dry-run origin main", true),
            ("git push --push-option nope --dry-run origin main", true),
            ("git push -nu origin main", true),
            ("git push --dry-run -of origin main", true),
            ("git push -o -f -n origin main", true),
            ("git push --repo=origin --dry-run origin main", true),
            ("git clean -e nope -n", true),
            ("git clean -e -f -n", true),
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
    /// `NAME=$(mktemp -d)` for unresolvable variables, and the help form for
    /// `cargo run` without bypass guidance.
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
            err.contains("--help"),
            "cargo run denial should suggest the help form: {err}"
        );
        for banned in ["target/debug", "built binary", "full shell", "escalate"] {
            assert!(
                !err.contains(banned),
                "cargo run denial must not teach a bypass: {err}"
            );
        }
    }

    #[test]
    fn symlink_escape_fails_closed() {
        // A symlink inside a temp root pointing at an existing non-temp dir:
        // writes through it land outside every temp root. The target must
        // exist for canonicalize to resolve through it; `/etc` is a stable
        // non-temp anchor that does not depend on the test process cwd (the
        // checkout can itself live under a temp root in CI/dev setups).
        let target = std::path::PathBuf::from("/etc");
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
            // Prefix execution options and quoted verbs route and track.
            ("cd /tmp && command -p cd /tmp && touch f", true),
            ("cd /tmp && time -p cd /tmp && touch f", true),
            ("cd /tmp && command \"cd\" /tmp && touch f", true),
            ("cd /tmp && eval 'cd /tmp' && touch f", true),
            ("cd /tmp && eval \"cd /tmp\" && touch f", true),
            // Informational/invalid prefix options never execute the cd, so
            // they are not routed — tracking stays where the real CWD is.
            ("cd /tmp && command -v cd /tmp && touch f", true),
            ("cd /tmp && builtin -p cd /tmp && touch f", true),
        ];

        run_cases(&cases);
    }

    #[test]
    fn quoted_eval_and_brace_escapes_fail_closed() {
        let cases = [
            // Quoted eval bodies run in the current shell: a cd inside moves
            // the real CWD behind the guard — fail closed.
            ("cd /tmp && eval 'cd /etc' && touch f", false),
            ("cd /tmp && eval \"cd /etc\" && touch f", false),
            ("cd /tmp && eval \"cd $HOME\" && touch f", false),
            ("cd /tmp && command eval 'cd /etc' && touch f", false),
            ("cd /tmp && time eval 'cd /etc' && touch f", false),
            ("cd /tmp && builtin eval 'cd /etc' && touch f", false),
            ("cd /tmp && eval 'cd /etc' && rm -rf x", false),
            // eval bodies with cd + extra commands reset fail-closed.
            ("cd /tmp && eval 'cd /etc; echo hi' && touch f", false),
            ("cd /tmp && eval 'cd /tmp && cd /etc' && touch f", false),
            // A `;` glued to the target is still a separator — the body is
            // not a simple cd, so tracking resets fail-closed.
            ("cd /tmp && eval 'cd /tmp; touch f' && touch g", false),
            // Quoted eval bodies WITHOUT a cd verb are the documented
            // eval-write residual — accepted (baseline behavior, pinned
            // non-goal; the guard does not model quoted eval body content).
            ("eval 'echo hi'", true),
            ("eval 'ls'", true),
            ("eval \"echo hi\"", true),
            ("eval 'touch /tmp/x'", true),
            ("eval 'rm -rf /tmp/x'", true),
            ("eval 'touch f'", true),
            ("eval 'echo hi && touch f'", true),
            ("eval 'cd /tmp; rm -rf /tmp/x'", true),
            // UNQUOTED eval bodies are visible and validated normally.
            ("eval echo hi", true),
            ("eval touch /tmp/x", true),
            ("eval touch f", false),
            ("eval rm -rf /__mahbot_readonly_test_ws__", false),
            // Brace groups run in the current shell: a cd inside changes the
            // real CWD without tracking, and a mutator inside writes
            // unvalidated — both fail closed.
            ("cd /tmp && { cd /etc; } && touch f", false),
            ("cd /tmp && { cd /etc } && touch f", false),
            ("cd /tmp && { cd /etc;\n} && touch f", false),
            ("cd /tmp && { cd /etc; } && rm -rf x", false),
            ("{ touch f; }", false),
            ("cd /tmp && { cd /tmp; } && touch f", false),
        ];

        run_cases(&cases);
    }

    #[test]
    fn prefix_option_and_quoted_verb_fail_closed() {
        let cases = [
            // `command -p` / `time -p` execute the cd in the current shell —
            // a non-temp target must reset fail-closed.
            ("cd /tmp && command -p cd /etc && touch f", false),
            ("cd /tmp && command -p cd -P /etc && touch f", false),
            ("cd /tmp && time -p cd /etc && touch f", false),
            ("cd /tmp && time -p cd ~ && touch f", false),
            (
                "cd /tmp && command -p cd /__mahbot_readonly_test_ws__ && rm -rf x",
                false,
            ),
            // Quoted cd/pushd verbs still execute in the current shell.
            ("cd /tmp && command \"cd\" /etc && touch f", false),
            ("cd /tmp && builtin 'cd' $HOME && touch f", false),
            ("cd /tmp && time \"cd\" /etc && touch f", false),
            ("cd /tmp && command 'pushd' /etc && touch f", false),
            ("\"cd\" /etc && touch f", false),
            (
                "cd /tmp && \"cd\" /__mahbot_readonly_test_ws__ && rm -rf x",
                false,
            ),
        ];

        run_cases(&cases);
    }

    /// Composed forwarding prefixes (`command builtin cd`, `time command cd`,
    /// `builtin -- command cd`, ...) execute the forwarded verb in the current
    /// shell — a cd hidden behind composed prefixes routes through cd tracking
    /// (fail-closed for non-temp targets) instead of falling through as an
    /// unmodeled non-mutator. Informational composed forms never execute.
    #[test]
    fn composed_forwarding_prefixes_fail_closed() {
        let cases = [
            // cd behind two forwarding prefixes — tracking updates/resets.
            ("cd /tmp && command builtin cd /etc && touch f", false),
            ("cd /tmp && command command cd /etc && touch f", false),
            ("cd /tmp && builtin command cd /etc && touch f", false),
            ("cd /tmp && command -p builtin cd /etc && touch f", false),
            ("cd /tmp && time command cd /etc && touch f", false),
            ("cd /tmp && time builtin cd /etc && touch f", false),
            ("cd /tmp && builtin -- command cd /etc && touch f", false),
            // Three levels.
            (
                "cd /tmp && command builtin command cd /etc && touch f",
                false,
            ),
            // Workspace-target variant.
            (
                "cd /tmp && command builtin cd /__mahbot_readonly_test_ws__ && rm -rf x",
                false,
            ),
            // Temp target behind composed prefixes tracks — write allowed.
            ("cd /tmp && command builtin cd /tmp && touch f", true),
            ("time command cd /tmp && touch f", true),
            // Combined -p flags still forward through the composition.
            ("cd /tmp && command -pp builtin cd /etc && touch f", false),
            // Informational composed forms never execute — tracking unchanged.
            ("cd /tmp && command -v builtin && touch f", true),
            ("cd /tmp && command builtin -v cd && touch f", true),
            ("cd /tmp && builtin command -v cd && touch f", true),
            ("cd /tmp && time command -v cd && touch f", true),
            // An env-assignment word after `command`/`builtin` is the COMMAND
            // name (`command FOO=bar cd /tmp` errors 127 — the cd never runs),
            // so tracking stays unchanged and a chained write fails closed.
            ("command FOO=bar cd /tmp; touch f", false),
            ("builtin FOO=bar cd /tmp; touch f", false),
            ("command -- FOO=bar cd /tmp; touch f", false),
            ("command -p FOO=bar cd /tmp; touch f", false),
            ("command -pp FOO=bar cd /tmp; touch f", false),
            ("command builtin FOO=bar cd /tmp; touch f", false),
            ("command -pp builtin FOO=bar cd /tmp; rm -rf x", false),
            ("command -p FOO=bar cd /tmp; rm -rf x", false),
            // `time` takes a full command list: the env assignment is valid
            // and the timed cd runs (tracked to the temp dir).
            ("time FOO=bar cd /tmp && touch f", true),
            ("time FOO=bar cd /etc && touch f", false),
            ("time -p FOO=bar cd /tmp && touch f", true),
        ];

        run_cases(&cases);
    }

    /// `time` is a forwarding prefix only as the UNQUOTED first word of a
    /// full segment, outside a `!`-negated list (bash 3.2 probe-verified):
    /// quoted `"time"`/`'time'`, an env assignment before it, a
    /// `command`/`builtin` before it, a nested `time time`, and `! time` all
    /// resolve to the external `time` command whose timed command runs in a
    /// child — the guard must not track a cwd the shell never reaches
    /// (fail-closed: a chained relative write against the workspace rejects).
    #[test]
    fn time_prefix_requires_head_position_and_unquoted() {
        let cases = [
            // Quoted `time` — P0 shape: the timed cd runs in a child, the
            // real CWD stays in the workspace, so a chained relative write
            // must reject against it.
            ("\"time\" cd /tmp && touch f", false),
            ("'time' cd /tmp && touch f", false),
            ("\"time\" cd /tmp; rm -rf x", false),
            // Bare quoted segment is allowed-but-untracked (nothing runs in
            // the workspace); P0 protection comes from the chained-write
            // rejection above.
            ("\"time\" cd /tmp", true),
            ("'time' cd /tmp", true),
            ("\"time\" cd /tmp && cd /tmp && touch f", true),
            // The shared-path quote-normalization still surfaces mutators
            // behind quoted `time` to the blocklist dispatch (external time
            // executes its command).
            ("\"time\" rm -rf /__mahbot_readonly_test_ws__", false),
            ("\"time\" git commit -m test", false),
            // The `cd /tmp && "time" cd /etc && touch f` shape flips from
            // REJECT to ALLOW: the first cd tracks /tmp, the quoted-time cd
            // runs in a child (no tracking change), and the write resolves
            // against the tracked /tmp — matching the real shell (the
            // pre-fix REJECT came from buggy /etc tracking).
            ("cd /tmp && \"time\" cd /etc && touch f", true),
            ("cd /tmp && 'time' cd /etc && touch f", true),
            // Composed prefixes demote `time` to the external command.
            ("command time cd /tmp && touch f", false),
            ("builtin time cd /tmp && touch f", false),
            ("command -- time cd /tmp && touch f", false),
            ("command time cd /etc && touch f", false),
            ("time time cd /tmp && touch f", false),
            ("command time export SNAP=/tmp; cd $SNAP; touch f", false),
            // Env assignment before `time` demotes it to external.
            ("TMPDIR=/tmp time cd /tmp && touch f", false),
            ("TMPDIR=/tmp time cd /etc && touch f", false),
            ("TMPDIR=/tmp time export SNAP=/tmp; touch $SNAP/f", false),
            (
                "TMPDIR=/tmp time export SNAP=/tmp; cd $SNAP; touch f",
                false,
            ),
            // `!` negation demotes `time` (external) but not `cd` (current
            // shell — the pinned `! cd /tmp && touch f` ALLOW stays).
            ("! time cd /tmp; touch f", false),
            ("! time cd /tmp; rm -rf x", false),
            ("! time cd /tmp && touch f", false),
            ("! time cd /tmp | cat; touch f", false),
            ("! cd /tmp; touch f", true),
            ("! cd /tmp && touch f", true),
            // Unquoted head-of-list `time` keeps forwarding (current-shell
            // timed command — tracking unchanged).
            ("time cd /tmp && touch f", true),
            ("time cd /etc && touch f", false),
            ("time -p cd /tmp && touch f", true),
            ("time command cd /tmp && touch f", true),
            ("time builtin cd /tmp && touch f", true),
            ("time export SNAP=/tmp; touch $SNAP/f", true),
            (
                "time export SNAP=/tmp && cd /tmp && cd $SNAP && touch f",
                true,
            ),
            ("time FOO=bar cd /tmp && touch f", true),
            ("time -p FOO=bar cd /tmp && touch f", true),
            // Quoted `-p` after unquoted head-of-list `time` is the TIMED
            // COMMAND (`-p: command not found` — nothing executes, no
            // tracking), not the option: a chained write lands in the
            // workspace and must reject. Bare segment is allowed-but-
            // untracked (the command errors, the real CWD is untouched).
            ("time \"-p\" cd /tmp; touch f", false),
            ("time \"-p\" cd /tmp; rm -rf x", false),
            ("time '-p' cd /tmp; touch f", false),
            ("time \"-p\" cd /tmp && touch f", false),
            ("time \"-p\" cd /tmp || touch f", false),
            ("time \"-p\" cd /tmp", true),
            // `'time' '-p'`: quoted time resolves external (child process),
            // and the quoted option would be its timed command either way —
            // no tracking, chained write rejects.
            ("'time' '-p' cd /tmp; touch f", false),
            // Unquoted `-p` option with a quoted timed command that errors.
            ("time -p \"-p\" cd /tmp; touch f", false),
            // Unquoted `-p` forwards a quoted timed command (`"cd"` is the
            // builtin — current shell, tracking unchanged).
            ("time -p \"cd\" /tmp && touch f", true),
        ];

        run_cases(&cases);
    }

    /// `time` is the EXTERNAL command at the first word of an
    /// if/elif/while/until condition or case-arm body (probe-verified on
    /// bash 3.2: the timed cd runs in a child, the parent CWD stays put) —
    /// the guard must not track a cwd the shell does not reach, so a write
    /// chained inside the condition rejects against the real CWD. A
    /// separator/newline before the first word returns `time` to the
    /// reserved word (`if \n time cd /tmp` tracks), and bodies keep it
    /// reserved.
    #[test]
    fn time_in_construct_conditions_is_external() {
        let cases = [
            // Condition-head / arm-head time: external — no tracking,
            // chained write rejects against the workspace.
            ("if time cd /tmp; touch f; then :; fi", false),
            ("if time cd /tmp && touch f; then :; fi", false),
            ("if time cd /tmp; rm -rf x; then :; fi", false),
            ("while time cd /tmp; touch f; do :; done", false),
            ("until time cd /tmp; touch f; do :; done", false),
            (
                "if false; then :; elif time cd /tmp; touch f; then :; fi",
                false,
            ),
            ("case x in x) time cd /tmp; touch f;; esac", false),
            // `!` after the keyword demotes too (already negated).
            ("if ! time cd /tmp; touch f; then :; fi", false),
            // A newline before the first word returns `time` to reserved:
            // tracked, chained temp write allowed.
            ("if\n time cd /tmp; touch f; then :; fi", true),
            // After a separator inside the condition / arm, `time` is
            // reserved again — the demotion covers the first unit only.
            ("if true && time cd /tmp && touch f; then :; fi", true),
            ("if cd /tmp && time cd /etc && touch f; then :; fi", false),
            ("case x in x) true && time cd /tmp && touch f;; esac", true),
            // Bodies keep `time` reserved (current-shell timed command).
            ("if true; then time cd /tmp; touch f; fi", true),
            ("if false; then :; else time cd /tmp; touch f; fi", true),
            ("while true; do time cd /tmp; touch f; break; done", true),
        ];

        run_cases(&cases);
    }

    // ── Fail-closed-by-default residuals (unmodeled constructs) ─────────

    /// Concatenated-quote / ANSI-C / escape / substitution-formed cd verbs
    /// reset (chained relative writes reject); the same verb forms for
    /// mutators reject outright. Each residual spelling gets its own case.
    #[test]
    fn unmodeled_verb_forms_fail_closed() {
        let cases = [
            // Concatenated-quote cd verbs — chained writes fail closed.
            ("cd /tmp && \"c\"d /etc && touch f", false),
            ("cd /tmp && \"c\"d /tmp && touch f", false),
            ("cd /tmp && c'd' /etc && touch f", false),
            ("cd /tmp && \"c\"'d' /etc && touch f", false),
            // ANSI-C quoting.
            ("cd /tmp && $'cd' /etc && touch f", false),
            ("cd /tmp && $'cd' /tmp && touch f", false),
            // Escape-formed verb.
            ("cd /tmp && c\\144 /etc && touch f", false),
            // Substitution-formed verb.
            ("cd /tmp && $(printf c)d /etc && touch f", false),
            ("cd /tmp && $(printf c)d /tmp && touch f", false),
            // Concatenated-quote pushd.
            ("cd /tmp && \"p\"ushd /etc && touch f", false),
            // Concatenated-quote mutators — rejected outright.
            ("\"r\"m -rf /__mahbot_readonly_test_ws__", false),
            ("\"t\"ouch /__mahbot_readonly_test_ws__/x", false),
            ("\"c\"p /etc/passwd /__mahbot_readonly_test_ws__/out", false),
            ("\"g\"it commit -m x", false),
            ("\"m\"kdir /__mahbot_readonly_test_ws__/x", false),
            ("$'rm' -rf /__mahbot_readonly_test_ws__", false),
            ("r\\155 -rf /__mahbot_readonly_test_ws__", false),
            ("$(printf r)m -rf /__mahbot_readonly_test_ws__", false),
            // Quoted mutators inside eval bodies — rejected outright.
            ("eval '\"t\"ouch /__mahbot_readonly_test_ws__/x'", false),
            ("eval '\"c\"d /etc' && touch f", false),
            // Fully-quoted mutators execute the unquoted command — the
            // blocklist dispatch normalizes the verb (item 2).
            ("\"rm\" -rf /__mahbot_readonly_test_ws__", false),
            ("\"touch\" /__mahbot_readonly_test_ws__/x", false),
            ("\"cp\" /etc/passwd /__mahbot_readonly_test_ws__/out", false),
            ("\"mkdir\" /__mahbot_readonly_test_ws__/x", false),
            ("\"sed\" -i s/a/b/ /__mahbot_readonly_test_ws__/f", false),
            ("\"git\" commit -m x", false),
            ("\"git\" status", true),
            // Prefix-wrapped quoted mutators route through the same
            // normalization (sudo/env/nice/npx forward the command).
            ("sudo \"rm\" -rf /__mahbot_readonly_test_ws__", false),
            ("env \"touch\" /__mahbot_readonly_test_ws__/x", false),
            (
                "nice \"cp\" /etc/passwd /__mahbot_readonly_test_ws__/out",
                false,
            ),
            ("npx \"git\" push", false),
            // Fully-quoted PREFIXES are stripped before the blocklist dispatch
            // too — `"env"` forwards like `env`, so the quoted mutator verb
            // behind it must still be found (quote-level-deep variant).
            ("\"env\" \"t\"ouch /__mahbot_readonly_test_ws__/x", false),
            ("\"exec\" \"r\"m -rf /__mahbot_readonly_test_ws__", false),
            ("\"npx\" \"g\"it push", false),
            ("\"env\" \"git\" push", false),
            ("\"nice\" \"t\"ouch /__mahbot_readonly_test_ws__/x", false),
            ("\"nohup\" \"t\"ouch /__mahbot_readonly_test_ws__/x", false),
            ("\"sudo\" \"rm\" -rf /__mahbot_readonly_test_ws__", false),
            ("\"env\" \"ls\" /tmp", true),
            ("\"env\" \"git\" status", true),
            // Brace-expansion verbs (`{touch,}` expands to `touch`) cannot be
            // normalized to a single word — unprovable, rejected (item 2).
            ("{touch,} f", false),
            ("{cp,} a b", false),
            ("{rm,} -rf /__mahbot_readonly_test_ws__", false),
            ("{touch,} /tmp/ok", false),
            ("cd /tmp && {touch,} f", false),
            ("{touch,} /tmp/x && echo hi", false),
            // Brace expansion in ARGUMENT position is not a verb — benign
            // commands with brace-expanded args stay allowed.
            ("echo {a,b}", true),
            ("ls {a,b}", true),
            // Fully-quoted benign verbs stay modeled and allowed.
            ("\"ls\" -la /tmp", true),
            ("\"echo\" hi", true),
            ("sudo \"ls\" /tmp", true),
            // Concatenated-quote benign verbs are unprovable — reject.
            ("\"c\"at /etc/passwd", false),
            // Fully-quoted cd stays modeled (balanced quotes are provable).
            ("cd /tmp && command \"cd\" /tmp && touch f", true),
            ("\"cd\" /tmp && touch f", true),
        ];

        run_cases(&cases);
    }

    /// `--` after a forwarding prefix executes the command in the current
    /// shell; a non-temp target must reset fail-closed. Informational/invalid
    /// prefix options never execute the command — tracking stays unchanged.
    #[test]
    fn prefix_dashdash_and_informational_options() {
        let cases = [
            ("cd /tmp && command -- cd /etc && touch f", false),
            ("cd /tmp && builtin -- cd /etc && touch f", false),
            ("cd /tmp && command -p -- cd /etc && touch f", false),
            ("cd /tmp && command -- 'cd' /etc && touch f", false),
            ("cd /tmp && command -- pushd /etc && touch f", false),
            // Forwarding forms with a temp target keep tracking.
            ("cd /tmp && command -- cd /tmp && touch f", true),
            ("cd /tmp && builtin -- cd -P /tmp && touch f", true),
            ("cd /tmp && command -p -- cd /tmp && touch f", true),
            // command -p is repeatable and forwards.
            ("cd /tmp && command -p -p cd /etc && touch f", false),
            ("cd /tmp && command -p -p cd /tmp && touch f", true),
            // Combined short flags forward like repeated -p (`-pp` executes);
            // a -v/-V anywhere in the combination is informational.
            ("cd /tmp && command -pp cd /etc && touch f", false),
            ("cd /tmp && command -pp cd /tmp && touch f", true),
            ("cd /tmp && command -ppp cd /tmp && touch f", true),
            ("command -pp rm -rf /__mahbot_readonly_test_ws__", false),
            ("cd /tmp && command -pp -v cd /etc && touch f", true),
            // Informational options never execute the command: tracking stays
            // where the real CWD is (a chained temp write stays allowed).
            ("cd /tmp && command -v cd /tmp && touch f", true),
            ("cd /tmp && command -V cd && touch f", true),
            ("cd /tmp && command -p -v cd /etc && touch f", true),
            ("cd /tmp && command -pv cd /etc && touch f", true),
            ("cd /tmp && command -x cd /etc && touch f", true),
            ("cd /tmp && builtin -p cd /tmp && touch f", true),
            // time consumes at most one -p: the second becomes the timed
            // command (`-p: command not found`), so cd never runs — tracking
            // unchanged. A chained relative write is validated against the
            // real (unchanged) CWD and must reject when that is the workspace.
            ("time -p -p cd /tmp && touch f", false),
            ("cd /tmp && time -p -p cd /etc && touch f", true),
            ("cd /tmp && time -p -p cd /tmp && touch f", true),
            // `--` after time is the timed command (not found) — informational.
            ("cd /tmp && time -- cd /etc && touch f", true),
            ("cd /tmp && time -v cd /etc && touch f", true),
            // time -p forwards.
            ("cd /tmp && time -p cd /etc && touch f", false),
        ];

        run_cases(&cases);
    }

    /// `cd` in a pipeline member or after a single `&` runs in a subshell or
    /// child — the parent CWD does not change, and the state before the
    /// boundary continues after it.
    #[test]
    fn pipeline_and_background_cd_fail_closed() {
        let cases = [
            // Pipeline member cd — parent CWD unchanged (workspace).
            ("cd /tmp | true && touch f", false),
            ("cd /tmp | cat && touch f", false),
            ("cd /tmp | true\n touch f", false),
            // State before the pipeline continues after it.
            ("cd /tmp && ls | grep x && touch f", true),
            ("cd /tmp && cd /etc | true && touch f", true),
            // Constructs as pipeline members run in subshells — their state
            // never leaks into the parent chain.
            ("ls | { SNAP=/tmp/x; } && touch $SNAP/f", false),
            ("ls | { cd /etc; } && touch f", false),
            ("ls | ( SNAP=/tmp/x ) && touch $SNAP/f", false),
            ("ls | if true; then SNAP=/tmp/x; fi && touch $SNAP/f", false),
            // A brace cd in a pipeline is contained (subshell) — the parent
            // cwd stays /tmp for the chained write.
            ("cd /tmp && ls | { cd /tmp; } && touch f", true),
            // Single-& background cd — parent CWD unchanged (workspace).
            ("cd /tmp & touch f", false),
            ("cd /tmp & rm -rf x", false),
            ("cd /tmp & touch /tmp/x", true),
            ("cd /tmp && touch a & touch f", false),
            ("cd /tmp & echo done", true),
            // `&&` is not a background separator.
            ("cd /tmp && touch f", true),
            // Redirect operators containing & are not separators.
            ("echo hi 2>&1", true),
            ("echo hi &> /tmp/out", true),
            ("echo hi >& /tmp/out", true),
            ("echo hi <&1", true),
            ("cat /tmp/x >& /tmp/out", true),
            // An ESCAPED `\>`/`\<` at segment end is a literal argument, not a
            // redirect operator — the following `&`/`|` IS a separator, so the
            // trailing command is validated independently (fail-closed).
            ("echo a\\> & touch f", false),
            ("echo a\\< & touch f", false),
            ("echo a\\> & touch /tmp/ok", true),
            ("echo a\\> | rm -rf /__mahbot_readonly_test_ws__", false),
            ("echo a\\> | grep x && touch /tmp/ok", true),
            ("echo a\\> 2>&1 & touch f", false),
            // A QUOTED `>`/`<` is a literal argument — the trailing quote keeps
            // the `&`/`|` a separator, so the trailing command is validated
            // independently (fail-closed).
            ("echo \">\" & touch f", false),
            ("echo '>' & touch f", false),
            ("echo \">\" | rm -rf /__mahbot_readonly_test_ws__", false),
            ("echo '<' | rm -rf /__mahbot_readonly_test_ws__", false),
            ("echo \"a>b\" & touch f", false),
            ("echo hi \"2>&1\" & touch f", false),
            ("echo x > \"/tmp/out\" & touch f", false),
            ("echo \"a\" > /tmp/out & touch f", false),
            // `>|` (clobber redirect) stays whole; `\>` before `|` does not.
            ("echo hi >| /tmp/out", true),
            // `||` is an OR-list, not a pipeline: the left side runs in the
            // current shell and the right side only on failure. State threads
            // like `&&` — a cd on the left tracks (baseline behavior).
            ("cd /tmp || touch f", true),
            ("cd /tmp || rm -rf x", true),
            ("cd /tmp || true && touch f", true),
            ("cd /etc || true && touch f", false),
            ("false || rm -rf /__mahbot_readonly_test_ws__", false),
            ("false || touch /__mahbot_readonly_test_ws__/x", false),
            // A subshell `||` is non-leaking — the parent cwd stays /tmp.
            ("cd /tmp && (cd /etc || true) && touch f", true),
            ("cd /tmp/nonexistent_x || touch f", false),
        ];

        run_cases(&cases);
    }

    /// Explicit PWD assignment must not be masked by tracked-cwd resolution.
    #[test]
    fn explicit_pwd_assignment_fails_closed() {
        let cases = [
            ("export PWD=/etc && touch $PWD/f", false),
            ("PWD=/etc touch $PWD/f", false),
            ("cd /tmp && export PWD=/etc && touch $PWD/f", false),
            ("cd /tmp && PWD=/etc touch $PWD/f", false),
            ("cd /tmp && export PWD=/etc && touch /tmp/ok", true),
            // A temp PWD binding resolves normally.
            ("cd /tmp && PWD=/tmp && touch $PWD/f", true),
            // Untracked PWD still resolves to the workspace root.
            ("touch $PWD/f", false),
            ("cd /tmp && touch $PWD/out", true),
        ];

        run_cases(&cases);
    }

    /// The brace-group eval shape (`{ eval 'cd /etc'; }`) runs the eval body
    /// in the current shell — chained relative writes fail closed.
    #[test]
    fn brace_eval_body_fails_closed() {
        let cases = [
            ("cd /tmp && { eval 'cd /etc'; } && touch f", false),
            ("cd /tmp && { eval \"cd /etc\"; } && touch f", false),
            ("cd /tmp && { eval 'cd /tmp'; } && touch f", false),
            ("cd /tmp && { eval 'cd /etc' ; } && rm -rf x", false),
            // A brace group with only reads stays allowed.
            ("cd /tmp && { ls; } && touch f", true),
            ("{ echo hi; } && touch /tmp/x", true),
            // A cd-SHAPED argument word is not a cd — no reset (position-aware).
            ("cd /tmp && { echo cd; } && touch f", true),
            ("cd /tmp && { echo \"hello cd\"; } && touch f", true),
            ("cd /tmp && { grep cd file; } && touch f", true),
        ];

        run_cases(&cases);
    }

    /// `cd` with extra operands: the runtime shell ignores extras and executes
    /// the cd to the first target — the guard cannot prove the target across
    /// shells, so tracking resets fail-closed.
    #[test]
    fn cd_extra_operands_fail_closed() {
        let cases = [
            ("cd /tmp extra && touch f", false),
            ("cd /tmp && cd /etc extra && touch f", false),
            ("cd /tmp && cd -P /tmp extra && touch f", false),
            ("cd /tmp && command cd /tmp extra && touch f", false),
            ("eval 'cd /tmp extra' && touch f", false),
        ];

        run_cases(&cases);
    }

    // ── Control constructs (fail-closed bodies/conditions, no leakage) ────

    /// Control-construct bodies and conditions get the same checks as
    /// top-level commands: workspace-targeting mutations and disallowed
    /// redirects reject; pure reads and temp-scoped writes stay allowed —
    /// across every construct class, both case spellings, nested constructs,
    /// constructs in substitutions, `!`-negated commands, and `&`/`|&`
    /// backgrounds.
    #[test]
    fn control_construct_bodies_and_conditions() {
        let cases = [
            // if / elif / else bodies and conditions.
            ("if true; then touch f; fi", false),
            ("if true; then touch /tmp/f; fi", true),
            ("if false; then echo hi; fi", true),
            ("if true; then echo hi > f; fi", false),
            ("if true; then echo hi > /tmp/out; fi", true),
            ("if cd /etc; then echo hi; fi", true),
            ("if true; then rm -rf f; else echo hi; fi", false),
            ("if false; then echo a; elif true; then touch f; fi", false),
            ("if false; then echo a; else touch /tmp/f; fi", true),
            ("if true\nthen touch f; fi", false),
            // while / until conditions and bodies.
            ("while true; do touch f; done", false),
            ("while true; do touch /tmp/f; done", true),
            ("while grep -q x file; do echo hi; done", true),
            ("until false; do rm -rf x; done", false),
            ("until false; do touch /tmp/f; done", true),
            // for: word lists, arithmetic headers, loop-var temp binding.
            ("for x in a b; do touch f; done", false),
            ("for x in a b; do touch /tmp/f; done", true),
            ("for x in /tmp/*.db; do rm \"$x\"; done", true),
            (
                "for db in ~/.mahbot/db/*.db; do cp \"$db\" /tmp/x/; done",
                true,
            ),
            ("for x in a; do rm \"$x\"; done", false),
            ("for ((i=0; i>3; i++)); do echo hi; done", true),
            ("for ((i=0; i>3; i++)); do touch f; done", false),
            ("for x; do echo hi; done", true),
            // `do` is a LIST WORD on the runtime shell (bash 3.2 iterates over
            // it in `for x in do; do ...`); omitting the `;`/newline before
            // `do` is a syntax error there, and the guard's rejection matches.
            ("for x in do; do echo hi; done", true),
            ("for x in a b do echo hi; done", false),
            // select.
            ("select x in a b; do touch f; done", false),
            ("select x in a b; do echo hi; done", true),
            // case: `|`-pattern and plain `;;` spellings.
            ("case x in a|b) touch f;; esac", false),
            ("case x in a|b) echo hi;; esac", true),
            ("case x in a) touch /tmp/f;; b) touch /tmp/g;; esac", true),
            ("case x in a) touch f;; esac", false),
            ("case x in\na) echo hi;;\nesac", true),
            ("case x in\nx) touch f;;\nesac", false),
            // subshells.
            ("( touch f )", false),
            ("( touch /tmp/f )", true),
            ("( cd /tmp && touch f )", true),
            ("( cd /etc && touch f )", false),
            // brace groups.
            ("{ touch f; }", false),
            ("{ touch /tmp/f; }", true),
            ("cd /tmp && { rm -rf x; }", true),
            // function definitions.
            ("f() { touch f; }", false),
            ("f() { touch /tmp/f; }", true),
            ("function f { rm -rf x; }", false),
            ("function f() { touch /tmp/f; }", true),
            ("function f () { touch /tmp/f; }", true),
            ("function f() { touch f; }", false),
            ("f () ( touch /tmp/f )", true),
            // nested constructs.
            ("if true; then for x in a; do touch f; done; fi", false),
            ("if true; then for x in a; do touch /tmp/f; done; fi", true),
            ("while true; do if false; then touch f; fi; done", false),
            (
                "if true; then while false; do cd /tmp; done; fi; touch f",
                false,
            ),
            // constructs in substitutions (subshell semantics).
            ("echo $(if true; then touch f; fi)", false),
            ("echo $(if true; then touch /tmp/f; fi)", true),
            ("echo $(for x in /tmp/*; do echo $x; done)", true),
            ("echo $(while true; do touch f; done)", false),
            // !-negated commands run in the current shell.
            ("! cd /tmp && touch f", true),
            ("! cd /etc && touch f", false),
            ("! ls && touch /tmp/x", true),
            // & and |& backgrounds.
            ("cd /tmp & touch f", false),
            ("cd /tmp & touch /tmp/x", true),
            ("cd /tmp |& true && touch f", false),
            ("ls |& grep x && touch /tmp/x", true),
            // Heredocs inside construct bodies: the body is stripped before
            // the walker; a workspace redirect on the delimiter line rejects
            // (the terminator must be on its own line — `EOF; fi` is not a
            // valid terminator in bash either, so nothing runs there).
            ("if true; then cat <<EOF\nbody\nEOF\nfi", true),
            ("if true; then cat <<EOF > f\nbody\nEOF\nfi", false),
            ("while true; do cat <<EOF > /tmp/out\nbody\nEOF\ndone", true),
            // Unterminated constructs reject.
            ("while true; do echo hi", false),
            ("for x in a b; do echo hi", false),
            ("{ echo hi; ", false),
            ("( echo hi ", false),
            // Header/pattern substitutions are still validated.
            ("for x in $(touch f); do echo hi; done", false),
            ("for x in $(touch /tmp/x); do echo hi; done", true),
            ("for ((i=$(touch f); i<3; i++)); do echo hi; done", false),
            ("case $(touch f) in a) echo hi;; esac", false),
            ("case $(touch /tmp/x) in a) echo hi;; esac", true),
            ("if $(touch f); then echo hi; fi", false),
            ("if $(touch /tmp/x); then echo hi; fi", true),
        ];

        run_cases(&cases);
    }

    /// Nothing set inside if/while/for/case/select bodies or subshells leaks
    /// after the construct: post-construct commands are validated from the
    /// pre-construct cwd — the `if false; then cd /tmp; fi; touch f` shape in
    /// both the `;` and newline spellings rejects.
    #[test]
    fn construct_state_does_not_leak() {
        let cases = [
            ("if false; then cd /tmp; fi; touch f", false),
            ("if false; then cd /tmp; fi\ntouch f", false),
            ("if true; then cd /tmp; fi; touch f", false),
            ("while false; do cd /tmp; done; touch f", false),
            ("for x in a; do cd /tmp; done; touch f", false),
            ("case x in a) cd /tmp;; esac; touch f", false),
            ("( cd /tmp ); touch f", false),
            ("f() { cd /tmp; }; touch f", false),
            // Pre-construct state threads INTO constructs.
            ("cd /tmp && if true; then touch f; fi", true),
            ("cd /tmp && while true; do touch f; done", true),
            ("cd /tmp && { touch f; }", true),
            (
                "SNAP=$(mktemp -d)\nif true; then touch \"$SNAP/f\"; fi",
                true,
            ),
            // Construct changes never affect a chained write after the
            // construct, even with an intervening separator.
            ("if false; then cd /tmp; fi ; touch f", false),
            // A cd inside a body or condition runs in the CURRENT shell (bash
            // if/for/while/case bodies are not subshells) — the leaked CWD is
            // untrackable, so chained relative writes fail closed.
            ("cd /tmp && if true; then cd /etc; fi && touch f", false),
            ("cd /tmp && if true; then cd /etc; fi\ntouch f", false),
            ("cd /tmp && if cd /etc; then :; fi && touch f", false),
            (
                "cd /tmp && if false; then :; elif true; then cd /etc; fi && touch f",
                false,
            ),
            (
                "cd /tmp && if false; then :; else cd /etc; fi && touch f",
                false,
            ),
            ("cd /tmp && for x in 1; do cd /etc; done && touch f", false),
            (
                "cd /tmp && while true; do cd /etc; break; done && touch f",
                false,
            ),
            (
                "cd /tmp && until false; do cd /etc; break; done && touch f",
                false,
            ),
            ("cd /tmp && case a in a) cd /etc;; esac && touch f", false),
            (
                "cd /tmp && if true; then if true; then cd /etc; fi; fi && touch f",
                false,
            ),
            // No cd in the construct: pre-construct tracking threads through.
            ("cd /tmp && if true; then echo hi; fi && touch f", true),
            ("cd /tmp && for x in 1; do echo hi; done && touch f", true),
            // Subshell / pipeline-member cds do NOT leak — state continues.
            ("cd /tmp && (cd /etc) && touch f", true),
            ("cd /tmp && (cd /tmp) && touch f", true),
            // Non-temp var rebindings inside current-shell construct bodies
            // leak at runtime (`if true; then export TMPDIR=/etc; fi; touch
            // $TMPDIR/f` writes to /etc) — the binding poisons the outer
            // tracking so the chained write fails closed.
            (
                "if true; then export TMPDIR=/etc; fi; touch $TMPDIR/f",
                false,
            ),
            (
                "if true; then export TMPDIR=/etc; fi\ntouch $TMPDIR/f",
                false,
            ),
            (
                "cd /tmp && if true; then export TMPDIR=/etc; fi && touch $TMPDIR/f",
                false,
            ),
            (
                "for x in 1; do export TMPDIR=/etc; done; touch $TMPDIR/f",
                false,
            ),
            (
                "while true; do export TMPDIR=/etc; break; done; touch $TMPDIR/f",
                false,
            ),
            (
                "case a in a) export TMPDIR=/etc;; esac; touch $TMPDIR/f",
                false,
            ),
            (
                "if true; then export TMPDIR=/etc; else export TMPDIR=/tmp; fi; touch $TMPDIR/f",
                false,
            ),
            // A temp-scoped rebind inside a body leaks but stays safe.
            (
                "if true; then export TMPDIR=/tmp; fi; touch $TMPDIR/f",
                true,
            ),
            // Subshell bodies do NOT leak their bindings.
            ("( export TMPDIR=/etc ); touch $TMPDIR/f", true),
            // A variable FIRST bound inside a conditional part is uncertain
            // (the branch may not execute, leaving it unset at runtime —
            // `if false; then D=/tmp; fi; touch $D/f` writes to /f), so it
            // poisons even when the binding is temp-scoped.
            ("if false; then D=/tmp; fi; touch $D/f", false),
            ("if true; then D=/tmp; fi; touch $D/f", false),
            ("if false; then D=/tmp; fi\ntouch $D/f", false),
            (
                "cd /tmp && if false; then D=/tmp; fi; cd $D && touch f",
                false,
            ),
            ("while false; do D=/tmp; done; touch $D/f", false),
            ("for x in 1; do D=/tmp; done; touch $D/f", false),
            ("case a in a) D=/tmp;; esac; touch $D/f", false),
            (
                "if true; then if false; then D=/tmp; fi; fi; touch $D/f",
                false,
            ),
            // Subshell bindings are discarded entirely — $D stays unbound.
            ("( D=/tmp ); touch $D/f", false),
        ];

        run_cases(&cases);
    }

    /// Quoted and prefix-composed `export` verbs bind exactly like a plain
    /// export (the shell concatenates quotes and the forwarding prefixes run
    /// in the current shell), so a non-temp binding poisons the tracked
    /// variable and a chained `$VAR` expansion fails closed. Non-forwarding
    /// prefixes (`env`/`sudo`) run the export in a child shell — nothing
    /// leaks, so the outer binding stays.
    #[test]
    fn quoted_and_prefixed_export_bindings() {
        let cases = [
            // Quoted export verbs.
            (
                "\"export\" TMPDIR=/__mahbot_readonly_test_ws__; touch $TMPDIR/f",
                false,
            ),
            ("'export' TMPDIR=/etc; touch $TMPDIR/f", false),
            // Quoted assignment words bind by their unquoted name.
            (
                "export \"TMPDIR=/__mahbot_readonly_test_ws__\"; touch $TMPDIR/f",
                false,
            ),
            ("export TMPDIR=\"/etc\"; touch $TMPDIR/f", false),
            ("export \"TMPDIR\"=/etc; touch $TMPDIR/f", false),
            // Forwarding prefixes that run export in the current shell.
            ("command export TMPDIR=/etc; touch $TMPDIR/f", false),
            ("builtin export TMPDIR=/etc; touch $TMPDIR/f", false),
            ("time export TMPDIR=/etc; touch $TMPDIR/f", false),
            ("command -p export TMPDIR=/etc; touch $TMPDIR/f", false),
            ("builtin -- export TMPDIR=/etc; touch $TMPDIR/f", false),
            ("command builtin export TMPDIR=/etc; touch $TMPDIR/f", false),
            ("time -p export TMPDIR=/etc; touch $TMPDIR/f", false),
            // Non-forwarding prefixes run export in a child — no leak.
            ("env export TMPDIR=/etc; touch $TMPDIR/f", true),
            ("sudo export TMPDIR=/etc; touch $TMPDIR/f", true),
            // Temp-target bindings through the same spellings stay allowed.
            ("\"export\" TMPDIR=/tmp; touch $TMPDIR/f", true),
            ("export \"TMPDIR=/tmp\"; touch $TMPDIR/f", true),
            ("command export TMPDIR=/tmp; touch $TMPDIR/f", true),
            ("\"export\" SNAP=/tmp/x; touch \"$SNAP/f\"", true),
            // Informational export forms never execute — no binding.
            ("command -v export; touch $TMPDIR/f", true),
            // Plain spellings (regression anchors).
            ("export TMPDIR=/etc; touch $TMPDIR/f", false),
            ("export TMPDIR=/tmp; touch $TMPDIR/f", true),
        ];

        run_cases(&cases);
    }

    /// The two pinned temp-idiot shapes stay allowed: a temp-cwd cd followed
    /// by a temp-scoped removal in a body, and a temp-dir binding followed by
    /// a temp-scoped copy in a body.
    #[test]
    fn pinned_temp_idiot_shapes_allowed() {
        let cases = [
            // temp-cwd cd followed by temp-scoped removal in a brace body.
            ("cd /tmp && { rm -rf x; }", true),
            ("cd /tmp && { rm -rf x; } && touch /tmp/ok", true),
            // temp-dir binding followed by temp-scoped copy in a brace body.
            ("SNAP=$(mktemp -d) && { cp /etc/passwd \"$SNAP/\"; }", true),
            (
                "SNAP=$(mktemp -d)\nmkdir -p \"$SNAP/d\"\n{ cp /etc/passwd \"$SNAP/d/\"; }",
                true,
            ),
            // The same shapes with workspace targets stay blocked.
            ("cd /tmp && { rm -rf /__mahbot_readonly_test_ws__; }", false),
            (
                "SNAP=$(mktemp -d) && { cp /etc/passwd /__mahbot_readonly_test_ws__/; }",
                false,
            ),
        ];

        run_cases(&cases);
    }

    /// Substitution-span matching balances every unquoted paren: nested
    /// arithmetic parens (`$(( (a) * (b) ))`), subshell parens, and nested
    /// `$(` substitutions all end at the correct close.
    #[test]
    fn substitution_end_nested_parens() {
        let s = "echo $(( (a) * (b) )) tail";
        let (content, next) = find_substitution_end(s, 7);
        assert_eq!(content, "( (a) * (b) )");
        assert_eq!(&s[next..], " tail");
        let s = "echo $( (echo hi) ) tail";
        let (_, next) = find_substitution_end(s, 7);
        assert_eq!(&s[next..], " tail");
        let s = "echo $(echo $(echo hi)) tail";
        let (_, next) = find_substitution_end(s, 7);
        assert_eq!(&s[next..], " tail");
    }
}
