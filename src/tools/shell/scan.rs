//! Frozen shell text-scanning helpers shared by out-of-scope consumers
//! (grep engine, output segmentation) and the read-only guard's keep-set
//! predicates.
//!
//! These helpers are **frozen**: the AST-based read-only guard does not rely
//! on them for command structure (heredoc stripping, quote handling, cd-option
//! policy, token classification, and substitution-aware word splitting all
//! serve consumers outside the guard), so they live here unchanged rather than
//! in the guard module. The guard still uses the word-level helpers
//! ([`split_words_keeping_substitutions`], [`strip_quoted_word`],
//! [`classify_shell_token`], ...) on words reconstructed from the syntax tree.

use super::track_char_context;

// ── Quote stripping ─────────────────────────────────────────────────────

/// Strip a balanced pair of surrounding quotes (`'...'` or `"..."`), returning
/// the inner content and whether the quotes were single. Returns `None` for
/// mixed (`'..."`), unbalanced, or interior-quote forms — callers fail closed.
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

/// Strip balanced surrounding quotes, keeping the raw word when unbalanced.
pub(super) fn strip_quoted_word(word: &str) -> &str {
    strip_outer_quotes(word).map_or(word, |(c, _)| c)
}

// ── cd option policy ────────────────────────────────────────────────────

/// Outcome of scanning a `cd`-family verb's argument words for its target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CdScan<'a> {
    /// Quote-stripped target plus the index of the word after it (the first
    /// extra-operand position).
    Target(&'a str, usize),
    /// No target after the options — a bare `cd`/`cd -P` (valid; targets $HOME).
    Bare,
    /// Option grammar violation (`-e`, `-@`, …) — the cd errors at runtime.
    BadOption,
}

/// Skip `cd` option flags (`-P`/`-L`/`--`, mirroring bash cd grammar) and
/// return the quote-stripped target plus the index after it. Option detection
/// uses the raw word (a quoted `"-P"` is a literal target); `-` alone is the
/// `$OLDPWD` target, never an option. `Bare` when no target follows;
/// `BadOption` when the option grammar is violated — call sites fail closed.
pub(super) fn cd_target_after_options<'a>(words: &[&'a str], start: usize) -> CdScan<'a> {
    let mut i = start;
    let mut options_ended = false;
    loop {
        let Some(w) = words.get(i) else {
            return CdScan::Bare;
        };
        if !options_ended && w.starts_with('-') && *w != "-" {
            if *w == "--" {
                options_ended = true;
            } else if !w[1..].bytes().all(|b| matches!(b, b'P' | b'L')) {
                return CdScan::BadOption;
            }
            i += 1;
            continue;
        }
        return CdScan::Target(strip_quoted_word(w), i + 1);
    }
}

// ── Heredoc body stripping ──────────────────────────────────────────────

/// Advance `i` past the current line (to the start of the next line or end of string).
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
/// redirect operators) to be removed from the scan string, making a later
/// redirect scan miss the redirect.
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
///   (a heredoc body containing a quoted delimiter line would end early).
/// - Multi-line engineered inputs are unlikely in practice.
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

        if !track_char_context(chars[i].1, &mut in_single, &mut in_double, &mut escaped) {
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
    let end = rest
        .find(|c: char| c.is_whitespace() || matches!(c, '<' | '>' | '&' | ';' | '|' | '(' | ')'))
        .unwrap_or(rest.len());
    (rest[..end].to_string(), start + end, false)
}

// ── Substitution spans ─────────────────────────────────────────────────

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
        if !track_char_context(c, &mut in_single, &mut in_double, &mut escaped) {
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

/// Find the matching `}` for a `${...}` parameter expansion whose `{` is at
/// byte `i + 1`. Bash closes at the first `}` not inside a nested
/// substitution span (verified against bash: `${x:-{a} rm f}` ends after
/// `{a`, so `rm f}` are separate words). Nested substitution spans are
/// skipped whole, so a `}` inside `$(...)`/backtick/`<(cmd)`/inner `${...}`
/// never closes the expansion early. The `}` check is quote-gated via
/// `track_char_context`: a `}` inside quotes is expansion content, so
/// `${x:-"a}b"}` closes at the final `}` — matching bash (quoted braces are
/// literal). Returns `(content, index_after_close)`; unterminated
/// expansions return the rest as content (validated anyway — fail-closed).
fn find_parameter_end(s: &str, i: usize) -> (&str, usize) {
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut k = i + 2;
    while k < s.len() {
        let c = s[k..].chars().next().expect("k < s.len()");
        if !escaped
            && !in_single
            && let Some((_, next)) = any_substitution_span(s, k, in_double)
        {
            k = next;
            continue;
        }
        if !track_char_context(c, &mut in_single, &mut in_double, &mut escaped) {
            k += c.len_utf8();
            continue;
        }
        if c == '}' {
            return (&s[i + 2..k], k + 1);
        }
        k += c.len_utf8();
    }
    (&s[i + 2..], s.len())
}

/// Span of a command substitution whose introducer is at byte `i` (`$(`,
/// backtick, or `${`). Returns `(content, index_after_close)` — `content`
/// starts AFTER the introducer (`i + 2` / `i + 1`), not at `i`; slice
/// `s[i..next]` for the full span.
pub(super) fn substitution_span(s: &str, i: usize) -> Option<(&str, usize)> {
    let b = s.as_bytes();
    if b.get(i) == Some(&b'$') && b.get(i + 1) == Some(&b'(') {
        Some(find_substitution_end(s, i + 2))
    } else if b.get(i) == Some(&b'`') {
        Some(find_backtick_end(s, i + 1))
    } else if b.get(i) == Some(&b'$') && b.get(i + 1) == Some(&b'{') {
        Some(find_parameter_end(s, i))
    } else {
        None
    }
}

/// `substitution_span` plus unquoted process substitution `<(cmd)`/`>(cmd)`.
/// `in_double` gates process substitution (bash keeps quoted process
/// substitution literal). Same `(content, index_after_close)` contract;
/// unterminated spans return the rest as content (fail-closed).
fn any_substitution_span(s: &str, i: usize, in_double: bool) -> Option<(&str, usize)> {
    if let Some(span) = substitution_span(s, i) {
        return Some(span);
    }
    let b = s.as_bytes();
    if !in_double
        && (b.get(i) == Some(&b'<') || b.get(i) == Some(&b'>'))
        && b.get(i + 1) == Some(&b'(')
    {
        let end = find_paren_close(s, i + 2, 1).unwrap_or(s.len());
        let content = if end == s.len() {
            &s[i + 2..]
        } else {
            &s[i + 2..end - 1]
        };
        return Some((content, end));
    }
    None
}

/// Walk `s` once with quote/escape context, invoking `visit(span, content,
/// end, in_double)` for every active substitution: `$(...)`, `` `...` ``, and
/// `${...}`, plus — only in unquoted words — process substitution
/// `<(cmd)`/`>(cmd)` (bash keeps quoted process substitution literal).
/// `span` is the full text incl. introducer and closing delimiter (or rest
/// when unterminated); `content` is the body without them; `end` the byte
/// index after the closing delimiter; `in_double` the double-quote state at
/// the span start (double-quoted expansions concatenate, unquoted ones
/// field-split — the fail-closed predicates need the distinction).
/// Single-quoted and escaped introducers are literal. Returning `false` from
/// `visit` stops the scan. This walk is the shared span-syntax scan: it backs
/// the guard's substitution predicates/decomposition
/// ([`crate::tools::shell::readonly::word_has_substitution`],
/// [`crate::tools::shell::readonly::substitution_could_form_flag`],
/// [`crate::tools::shell::readonly::word_parts`]), so span syntax cannot
/// drift between them. The word tokenizer ([`split_words_keeping_substitutions`])
/// and the bare-`$var` detector walk separately — they traverse for word
/// boundaries and unbraced `$` starts, not for spans.
pub(super) fn for_each_substitution(
    s: &str,
    mut visit: impl FnMut(&str, &str, usize, bool) -> bool,
) {
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut i = 0;
    while i < s.len() {
        let c = s[i..].chars().next().expect("i < s.len()");
        // Detect substitution starts before the quote-state skip: they run
        // inside double quotes (`"$(...)"`, `` "`...`" ``) but are literal
        // inside single quotes or after an escape backslash.
        if !escaped
            && !in_single
            && let Some((content, next)) = any_substitution_span(s, i, in_double)
        {
            if !visit(&s[i..next], content, next, in_double) {
                return;
            }
            i = next;
            continue;
        }
        if !track_char_context(c, &mut in_single, &mut in_double, &mut escaped) {
            i += c.len_utf8();
            continue;
        }
        i += c.len_utf8();
    }
}

// ── Token classification ────────────────────────────────────────────────

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
/// checks below (the `w.starts_with('>')` arm would misclassify `>&` as a
/// self-contained combined token and `&>`/`&>>` as regular words).
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

// ── Substitution-aware word splitter ───────────────────────────────────

/// Split a shell command string into words, keeping command-substitution
/// spans (`$(...)`, backticks, `${...}`, unquoted `<(cmd)`/`>(cmd)`) whole
/// even when they contain whitespace or shell operators — bash treats each
/// span as part of ONE word. Quote- and escape-aware: whitespace inside
/// quotes is not a word boundary.
pub(super) fn split_words_keeping_substitutions(s: &str) -> Vec<&str> {
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
            && let Some((_, next)) = any_substitution_span(s, i, in_double)
        {
            i = next;
            continue;
        }
        if !track_char_context(c, &mut in_single, &mut in_double, &mut escaped) {
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
