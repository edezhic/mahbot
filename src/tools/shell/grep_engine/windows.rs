//! Windows (`cmd.exe`) command model of the transparent grep interception.
//!
//! Compiled on every platform and driven by the runtime platform value
//! ([`crate::tools::shell::SHELL_PLATFORM`] — one `if cfg!(windows)` constant
//! that every consumer here takes as data, never as a `cfg` branch of its own),
//! the way the read-only guard's own `windows` layer is: the parent reads a
//! command string exactly the way `cmd.exe /C` would split, quote and dispatch
//! it, so the whole model is unit-testable from any host.
//!
//! `sh -c` and `cmd.exe /C` disagree on nearly every token. `&` separates
//! commands for cmd and backgrounds a job for sh; `;` separates for sh and is
//! ordinary text for cmd (several of its own internal commands take it as an
//! argument delimiter); `^` escapes the next character and `(...)` groups
//! commands for cmd; only `"` quotes, with `""` as a literal quote inside a
//! quoted span, while `'` and `\` are ordinary characters. The unix segmenter,
//! tokenizer and matcher are therefore never run on a Windows command string —
//! every decision below is the cmd.exe reading, and everything the model cannot
//! read is refused fail-closed.
//!
//! # Spec hand-off
//!
//! cmd.exe's command line is capped at 8191 characters and the shell re-parses
//! the whole line (`%name%` expansion, separators, carets) before the program
//! sees its argv, so no quoting carries a 64 KiB JSON payload intact. A served
//! member therefore passes its spec through a scratch file under the daemon's
//! private temp root ([`scratch_dir`], which is the process test root in the
//! host lane) and hands the path over with [`SPEC_FILE_FLAG`]; the parent
//! deletes the file when the call finishes and the temp cleaner is the backstop
//! for a killed daemon.
//!
//! The rewrite reaches cmd.exe through the shell's own spawn
//! (`crate::tools::shell::build_shell_command`), which passes it as the `/C`
//! argument in the extra quote pair cmd.exe's own quote processing strips —
//! `raw_arg`, not `arg`, because std's `CommandLineToArgvW`-style escaping is
//! not cmd.exe's reading and would mangle the rewrite's own quotes.
//!
//! The one machine-wide limit this hand-off keeps: the rewrite names two paths
//! this process did not choose — the running executable's installation path and,
//! through it, the temp root the scratch file sits under. [`cmd_quote`] refuses
//! a `%`, `!` or `"` in either, because cmd.exe re-reads those even inside
//! quotes and a mangled path breaks the hand-off itself, so every search on such
//! a machine fails (loudly, never silently) until the path changes. The narrower
//! reading applied to the agent's own search text (two or more `%` characters
//! in one word, see [`has_percent_expansion`]) answers a different question:
//! that text rides the spec file and never appears on the command line.
//!
//! # Path identity
//!
//! The parent's tracked cwd and the engine's own `current_dir()` both go
//! through [`super::canonical_or_lexical`], so the cwd gate only then needs
//! [`same_directory`]'s case/separator/verbatim-insensitive comparison rather
//! than byte equality; the same canonicalization keeps displayed and operand
//! paths in the platform's natural spelling (it strips the verbatim prefix).
//!
//! `readonly/windows.rs` reads cmd.exe with its own reader and its own
//! fail-closed policy, so a cmd.exe rule change belongs in both places.
//!
//! # Unverifiable from this host
//!
//! No Windows host or CI is available to this crate, so cmd.exe's own reading
//! is argued rather than measured: that it splits a line on `&`/`|`/newlines,
//! keeps a `>`-family redirect's own `&` inside the redirect (`2>&1`), treats
//! `;`/`'`/`\` as ordinary characters, reads `""` inside a quoted span as a
//! literal quote, dispatches verbs case-insensitively and extension-insensitively
//! (`GREP.EXE`), changes nothing on a bare `cd` while recognising `/d` as
//! its only switch, and keeps the directory stack `pushd` pushes and `popd`
//! pops. Every refusal rests on that reading — as does the reading
//! of the paths this layer joins and compares, and the `/C "…"` hand-off the
//! shell spawns a rewrite with. The one decision with a host oracle,
//! [`fnmatch`], is pinned differentially against the host's own `fnmatch` by
//! the parent module's `windows_fnmatch_parity` tests — a unix host's pin: a
//! Windows test host has no second implementation to compare against. One
//! spelling stays unmeasurable even so: a newline inside a quoted verbatim
//! member is not a connector, so it stays in the rewrite and is the
//! newline-in-`/C` spelling nothing here can measure (the segmenter's newline
//! connector is re-emitted as `&` for exactly this reason — see
//! [`super::join_rewritten`]).

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::tools::shell::scan::{self, CdScan};

use super::GrepWord;

/// The argument a served member receives its spec path through.
pub(super) const SPEC_FILE_FLAG: &str = "--spec-file";

/// Scratch directory (below the daemon's private temp root) holding the spec
/// files of served Windows members.
const SPEC_DIR: &str = "grep-engine";

// ── Command splitting ────────────────────────────────────────────────────

/// The `&` at the cursor belongs to the redirect token spelled just before it
/// (`2>&1`, `>&2`, `>& file`): a `&` directly after `>` merges descriptors
/// instead of chaining commands, so it never ends the member.
///
/// Reading the member's last character is a spelling test, not a token read: a
/// word-glued redirect whose target meets the next command (`grep a>&echo done`)
/// is one member here while cmd.exe sees a `>&` redirect plus `echo`. The glued
/// word itself is then refused as unreadable ([`has_glued_redirect`]), so the
/// divergence never produces a search of different text.
fn redirect_keeps_amp(before: &str) -> bool {
    before.ends_with('>')
}

/// Push the trimmed member of `current` onto `out` with its connector.
/// `required` marks a member that must not be empty (one before a `&`/`|`/
/// `&&`/`||` connector — a shape cmd.exe reads as an extra command, which
/// dropping silently would turn into a valid single command).
fn flush(
    current: &mut String,
    out: &mut Vec<(String, String)>,
    conn: &str,
    required: bool,
) -> bool {
    let text = current.trim().to_string();
    current.clear();
    if text.is_empty() {
        return !required;
    }
    out.push((text, conn.to_string()));
    true
}

/// Split a command line the way `cmd.exe /C` does: on `&`, `&&`, `|`, `||` and
/// newlines, returning `(segment, following-connector)` pairs with `""` as the
/// last connector. `None` for a line whose reading is not modelled:
///
/// - a `^` or a `(`/`)` outside double quotes: the caret escapes the next
///   character and the parentheses group commands, so a member split at the
///   wrong place could run a command the agent did not write;
/// - an unbalanced `"`;
/// - an empty member before a connector (`&& a`) or a trailing connector with
///   no member after it (`a &&`, `a |`) — the unix segmenter's fail-closed
///   policy, mirrored.
///
/// `;` is ordinary text here (cmd.exe keeps one line) and is never a
/// connector; the caller reads it through [`unquoted_semicolon`]. Blank lines
/// and a trailing newline stay valid, exactly as in the unix segmenter.
#[must_use]
pub(super) fn segment_command(command: &str) -> Option<Vec<(String, String)>> {
    let chars: Vec<char> = command.chars().collect();
    let mut out: Vec<(String, String)> = Vec::new();
    let mut current = String::new();
    let mut in_double = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '"' {
            // `""` inside a quoted span is a literal quote, not a close+open.
            if in_double && chars.get(i + 1) == Some(&'"') {
                current.push_str("\"\"");
                i += 2;
                continue;
            }
            in_double = !in_double;
            current.push('"');
            i += 1;
            continue;
        }
        if in_double {
            current.push(c);
            i += 1;
            continue;
        }
        match c {
            // Unmodellable outside quotes (see the doc comment).
            '^' | '(' | ')' => return None,
            '\n' => {
                flush(&mut current, &mut out, "\n", false);
                i += 1;
            }
            '&' | '|' => {
                if c == '&' && redirect_keeps_amp(&current) {
                    current.push('&');
                    i += 1;
                    continue;
                }
                let doubled = chars.get(i + 1) == Some(&c);
                let conn = match (c, doubled) {
                    ('&', true) => "&&",
                    ('&', false) => "&",
                    ('|', true) => "||",
                    ('|', false) => "|",
                    _ => unreachable!("only `&` and `|` reach this arm"),
                };
                if !flush(&mut current, &mut out, conn, true) {
                    return None;
                }
                i += if doubled { 2 } else { 1 };
            }
            _ => {
                current.push(c);
                i += 1;
            }
        }
    }
    if in_double {
        return None;
    }
    flush(&mut current, &mut out, "", false);
    // A trailing connector leaves cmd.exe with nothing after it — the same
    // fail-closed policy as the unix segmenter's trailing-pipe check.
    if matches!(
        out.last().map(|(_, c)| c.as_str()),
        Some("&" | "&&" | "|" | "||")
    ) {
        return None;
    }
    Some(out)
}

/// True when a `;` appears outside double quotes. cmd.exe does not split a
/// command on it, so a caller must never read the fragments around one as
/// separate commands.
#[must_use]
pub(super) fn unquoted_semicolon(command: &str) -> bool {
    let mut in_double = false;
    let mut chars = command.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '"' {
            if in_double && chars.peek() == Some(&'"') {
                chars.next();
                continue;
            }
            in_double = !in_double;
            continue;
        }
        if !in_double && c == ';' {
            return true;
        }
    }
    false
}

// ── Verb and word reading ────────────────────────────────────────────────

/// The delivered text of a word: the inner text of a balanced pair of
/// surrounding double quotes, or the word itself when it carries no quote
/// character at all. `None` for a spelling with an interior or unbalanced `"`
/// — cmd reads those differently from the word as written, so callers fail
/// closed rather than judge characters that are not the ones delivered.
fn strip_double_quotes(word: &str) -> Option<&str> {
    if let Some(inner) = word
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
    {
        return (!inner.contains('"')).then_some(inner);
    }
    (!word.contains('"')).then_some(word)
}

/// The key cmd.exe's own dispatch matches a verb word on: its delivered text,
/// lower-cased, with a trailing `.exe` folded away — `GREP`, `Grep.EXE` and
/// `grep` are one verb. Every verb list in the engine (grep family, `cd`
/// family, command introducers, the programs that own their search) is written
/// in that spelling and read through this key, so a verb is classified
/// identically however the interpreter and its switch are spelled. `None` for a
/// spelling whose delivered text is not the word as written (an interior or
/// unbalanced `"`): the shell would re-read it, so no list may be matched
/// against it.
#[must_use]
pub(super) fn verb_key(word: &str) -> Option<String> {
    let word = strip_double_quotes(word)?;
    // Fold a `.exe` suffix in any case (`Grep.EXE`), one ASCII extension only.
    let bytes = word.as_bytes();
    let stem = if bytes.len() >= 4 && bytes[bytes.len() - 4..].eq_ignore_ascii_case(b".exe") {
        &word[..word.len() - 4]
    } else {
        word
    };
    Some(stem.to_ascii_lowercase())
}

/// The grep family a command word names, or `None` when it names something
/// else. cmd.exe dispatches case-insensitively and through `PATHEXT`, so
/// `GREP`, `Grep.EXE` and `grep` all run the same program; only `.exe` is
/// folded, since the other executable extensions (`.com`, `.bat`, `.cmd`) name
/// a program of their own rather than grep. A path-qualified word names a file
/// rather than a shell dispatch, so it is not read as this platform's grep at
/// all and the segment is left to the platform as written — the reading unix
/// gives `./grep`, which is not the `grep` verb either.
#[must_use]
pub(super) fn grep_verb(word: &str) -> Option<&'static str> {
    let key = verb_key(word)?;
    if key.is_empty() || key.contains(['\\', '/']) {
        return None;
    }
    match key.as_str() {
        "grep" => Some("grep"),
        "egrep" => Some("egrep"),
        "fgrep" => Some("fgrep"),
        _ => None,
    }
}

/// Push the accumulated token onto `out`, classifying it through the shared
/// redirect classifier (the single source of redirect-token semantics).
fn flush_token(raw: &mut String, value: &mut String, out: &mut Vec<GrepWord>) {
    if raw.is_empty() {
        return;
    }
    let (redirect, needs_target) = match scan::classify_shell_token(raw) {
        scan::TokenKind::Regular => (false, false),
        scan::TokenKind::Redirect { needs_target } => (true, needs_target),
    };
    out.push(GrepWord {
        value: std::mem::take(value),
        raw: std::mem::take(raw),
        redirect,
        needs_target,
    });
}

/// cmd.exe's own word splitting for one member: whitespace-separated,
/// `"`-quoting with `""` as a literal quote inside a quoted span, and — unlike
/// `sh` — no escape character at all, so a backslash is ordinary text and `'`
/// does not quote. `None` on an unbalanced quote.
///
/// The value keeps `$` and backticks literal: cmd.exe expands neither, and `$`
/// is a common ERE anchor. The `%` reading is not this function's: cmd.exe
/// expands `%…%` before the program sees its argv, so the caller fails
/// closed on the whole member (see [`has_percent_expansion`]).
#[must_use]
pub(super) fn tokenize(segment: &str) -> Option<Vec<GrepWord>> {
    let mut out = Vec::new();
    let mut raw = String::new();
    let mut value = String::new();
    let mut in_double = false;
    let mut chars = segment.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '"' {
            if in_double && chars.peek() == Some(&'"') {
                chars.next();
                raw.push_str("\"\"");
                value.push('"');
                continue;
            }
            in_double = !in_double;
            raw.push('"');
            continue;
        }
        if !in_double && c.is_whitespace() {
            flush_token(&mut raw, &mut value, &mut out);
            continue;
        }
        raw.push(c);
        value.push(c);
    }
    if in_double {
        return None;
    }
    flush_token(&mut raw, &mut value, &mut out);
    Some(out)
}

/// True when `raw` carries an unquoted redirect operator after its first
/// character — the glued spelling cmd.exe splits into an argument plus a
/// redirect (`x>out.txt` is the argument `x` and a redirect to `out.txt`).
/// Words that *open* with an operator are [`scan::classify_shell_token`]'s
/// redirects already; this covers the form the shared classifier reads as one
/// ordinary word, which the model here must not read differently from the
/// interpreter (see `grep_tokenize`).
#[must_use]
pub(super) fn has_glued_redirect(raw: &str) -> bool {
    let mut in_quotes = false;
    for (at, c) in raw.char_indices() {
        match c {
            '"' => in_quotes = !in_quotes,
            '<' | '>' if !in_quotes && at > 0 => return true,
            _ => {}
        }
    }
    false
}

/// Unquote one raw word the way cmd.exe delivers it: balanced double quotes
/// removed, `""` inside the quoted span read as a literal quote, and every
/// other character literal — no expansion, no escape. Idempotent on the value
/// [`tokenize`] produced, which is what the operand path passes here.
#[must_use]
pub(super) fn unquote_word(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut in_double = false;
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '"' {
            if in_double && chars.peek() == Some(&'"') {
                chars.next();
                out.push('"');
                continue;
            }
            in_double = !in_double;
            continue;
        }
        out.push(c);
    }
    out
}

/// Quote one argument for a cmd.exe command line.
///
/// cmd.exe re-reads the whole line before the program sees its argv, but it
/// stops treating `&`, `|`, `<`, `>`, `^`, `(`, `)`, `;` and `=` as anything
/// but text inside a double-quoted argument — which is what makes a path like
/// `C:\Program Files (x86)\…` servable at all. What cmd rewrites even inside
/// quotes is `%` (always) and `!` (with delayed expansion), plus the quote
/// itself and a line terminator: those cannot be passed literally, so refusing
/// the word is fail-closed. Both callers pass a path this process did not choose
/// — the executable's installation path, and the scratch file under the temp
/// root — so a path carrying one of them is the per-machine condition the module
/// header states rather than a per-command one.
pub(super) fn cmd_quote(text: &str) -> Result<String, String> {
    let forbidden = ['%', '!', '"', '\n', '\r'];
    if let Some(bad) = text.chars().find(|c| forbidden.contains(c)) {
        return Err(format!("argument contains `{bad}`"));
    }
    Ok(format!("\"{text}\""))
}

/// Allocate the scratch path one served member's spec is written to:
/// `<scratch dir>/<pid>-<n>.json`, the counter making the name unique within
/// the process. Pure path arithmetic: the directory is [`write_spec_file`]'s to
/// create and the file is the caller's to remove when the call finishes. The
/// path is quoted before anything is written and goes onto that removal list
/// before the write, so neither a refused argument nor a failed write leaves a
/// file behind. See the module header for why the spec cannot ride on the
/// command line.
pub(super) fn spec_file_path() -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    scratch_dir().join(format!(
        "{}-{}.json",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ))
}

/// The directory [`spec_file_path`] allocates in: [`crate::temp::shell_tmpdir`]
/// — the daemon's pinned private root — and, under `cfg(test)`, the
/// process-level test root instead. A test run must never create or delete
/// inside a temp root a running daemon owns (the lane inherits its environment),
/// and the daemon's cleaner is not the test lane's backstop; that root is also
/// the lane's fake storage root, so the scratch files sit in their own
/// `grep-engine` subdirectory and go away with the root at process exit on unix
/// (elsewhere the OS temp sweep reclaims them).
fn scratch_dir() -> PathBuf {
    #[cfg(test)]
    {
        crate::util::test::test_root().join(SPEC_DIR)
    }
    #[cfg(not(test))]
    {
        PathBuf::from(crate::temp::shell_tmpdir()).join(SPEC_DIR)
    }
}

/// Write one engine spec to its scratch path, creating the scratch directory if
/// it is not there yet. The file is created fresh: `create_new` refuses to
/// follow a symlink planted at the name, so a planted link cannot redirect this
/// write; a leftover file from an earlier run is removed first so it cannot
/// turn the write into a refusal, though an entry that cannot be removed at all
/// (a directory, an ACL-locked file) does refuse the member. The error is a
/// rendered location for the fallback reason, never an errno the caller matches
/// on: the member is refused rather than run without a spec it cannot be served
/// from.
pub(super) fn write_spec_file(path: &Path, json: &str) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let _ = fs::remove_file(path);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    file.write_all(json.as_bytes())
        .map_err(|e| format!("{}: {e}", path.display()))
}

// ── Windows path reading ─────────────────────────────────────────────────

/// True when `word` is an absolute Windows spelling: a drive root (`C:\`,
/// `C:/`), a UNC server/share (`\\srv\share`, also `//srv/share`), or a
/// separator-rooted path (`\x`, `/x`). A drive-RELATIVE spelling (`C:x`) is
/// not absolute.
#[must_use]
pub(super) fn is_absolute_word(word: &str) -> bool {
    word.starts_with(['\\', '/']) || drive_anchor(word).is_some()
}

/// Join a display path: with `\` when the prefix carries no trailing
/// separator, directly when it does (the prefix of [`split_components`] always
/// does, and keeps the spelling it was typed with).
#[must_use]
pub(super) fn join_display(prefix: &str, name: &str) -> String {
    if prefix.is_empty() || prefix.ends_with(['\\', '/']) {
        format!("{prefix}{name}")
    } else {
        format!("{prefix}\\{name}")
    }
}

/// The display path of a walked entry: the operand's own spelling with the
/// entry's relative suffix [`join_display`]ed on. A root that is nothing but a
/// separator (`\`, `/`) keeps exactly one, and an empty root contributes none.
#[must_use]
pub(super) fn traversal_display(root_display: &str, rel: &str) -> String {
    let root = root_display.trim_end_matches(['\\', '/']);
    let root = if root.is_empty() {
        root_display.get(..1).unwrap_or("")
    } else {
        root
    };
    join_display(root, rel)
}

/// The `\\server\share\` anchor of a UNC pattern, including its trailing
/// separator and in the spelling it was typed with. `None` when the pattern is
/// not UNC or names no server/share (a degenerate anchor has no root to walk).
fn unc_anchor(pattern: &str) -> Option<&str> {
    let rest = pattern
        .strip_prefix("\\\\")
        .or_else(|| pattern.strip_prefix("//"))?;
    let mut parts = rest.splitn(3, ['\\', '/']);
    let server = parts.next().filter(|s| !s.is_empty())?;
    let share = parts.next().filter(|s| !s.is_empty())?;
    parts.next()?; // no component path below the anchor
    // The anchor ends after the separator following the share.
    let len = server.len() + share.len() + 4;
    Some(&pattern[..len])
}

/// The `:` of a leading drive spelling (`C:`, `C:\`, `C:foo`) — the one place
/// the drive-letter test lives, read by both the drive-rooted and the
/// drive-relative reader below. `None` for a word that names no drive.
fn drive_colon(word: &str) -> Option<usize> {
    let bytes = word.as_bytes();
    (bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':').then_some(1)
}

/// The `C:\`/`C:/` anchor of a drive-rooted pattern, including its trailing
/// separator. `None` for a drive-RELATIVE spelling (`C:foo`), which has no
/// root to walk.
fn drive_anchor(pattern: &str) -> Option<&str> {
    let colon = drive_colon(pattern)?;
    if !matches!(pattern.as_bytes().get(colon + 1), Some(b'\\' | b'/')) {
        return None;
    }
    Some(&pattern[..colon + 2])
}

/// Split a glob pattern into its walk-root prefix and the components below it,
/// at `\` and `/` boundaries. The prefix keeps the anchor Windows resolves
/// first — a drive (`C:\`), a UNC server/share (`\\srv\share\`) or a rooted
/// separator (`\`, `/`) — and always ends with the separator it was typed with;
/// every later separator is a component boundary. `None` for a spelling with no
/// root to walk (a drive-relative `C:foo`, a server- or share-less `\\srv`) or
/// with no component left to match.
#[must_use]
pub(super) fn split_components(pattern: &str) -> Option<(String, Vec<String>)> {
    // A drive-relative spelling (`C:foo`) names the cwd of ANOTHER drive: it has
    // no root to walk, so it is refused rather than resolved against this one.
    if drive_colon(pattern).is_some() && drive_anchor(pattern).is_none() {
        return None;
    }
    let (prefix, rest) = if pattern.starts_with("\\\\") || pattern.starts_with("//") {
        // A `\\`/`//`-leading spelling is a UNC (or device) path: without a
        // readable server/share anchor there is no root to walk, and reading it
        // as a rooted separator would resolve a different directory than cmd.
        let anchor = unc_anchor(pattern)?;
        (anchor.to_string(), &pattern[anchor.len()..])
    } else if pattern.starts_with(['\\', '/']) {
        (pattern[..1].to_string(), &pattern[1..])
    } else if let Some(anchor) = drive_anchor(pattern) {
        (anchor.to_string(), &pattern[anchor.len()..])
    } else {
        (String::new(), pattern)
    };
    let comps: Vec<String> = rest
        .split(['\\', '/'])
        .filter(|c| !c.is_empty())
        .map(str::to_string)
        .collect();
    if comps.is_empty() {
        return None;
    }
    Some((prefix, comps))
}

/// True when the token carries a glob metacharacter (`*`, `?`, `[`) outside
/// double quotes — cmd.exe expands nothing, so an unquoted metacharacter is a
/// glob the program reading the operand expands.
#[must_use]
pub(super) fn has_unquoted_glob(tok: &str) -> bool {
    let mut in_double = false;
    let mut chars = tok.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '"' {
            if in_double && chars.peek() == Some(&'"') {
                chars.next();
                continue;
            }
            in_double = !in_double;
            continue;
        }
        if !in_double && matches!(c, '*' | '?' | '[') {
            return true;
        }
    }
    false
}

/// True when the word holds two or more `%` characters — the reading both
/// callers approximate cmd.exe's `%…%` expansion with. cmd.exe pairs its
/// percents across the whole line, which this does not model: the test is
/// per-word, and two `%` characters anywhere in one word are enough, so `100%%`
/// is refused too. That is enough here because the served member's words never
/// ride the command line — they go to the spec file — so the only expansion that
/// can matter is one the engine would misread; a `cd` target does stay verbatim
/// in the rewrite and is refused on the same approximation (over-refusing rather
/// than mis-tracking). An undefined name expands to nothing, so the text between
/// the percents is never what the program receives. A lone `%` is ordinary text,
/// and so is every `!`: this process spawns `cmd /C` without `/V:ON`, so delayed
/// expansion is off and `!name!` is delivered literally (a machine registry
/// default could enable it — accepted, it is the literal reading the agent
/// wrote).
#[must_use]
pub(super) fn has_percent_expansion(word: &str) -> bool {
    word.matches('%').count() >= 2
}

/// Windows identity of two directories, for the engine's cwd gate: case,
/// separator spelling and a verbatim prefix are all insignificant there, and
/// cmd.exe's own cwd never spells a directory the way `fs::canonicalize` does.
/// Byte equality is deliberately not the test.
#[must_use]
pub(super) fn same_directory(a: &Path, b: &Path) -> bool {
    dir_key(a) == dir_key(b)
}

/// [`same_directory`]'s comparison form: verbatim prefix stripped, `/` folded
/// to `\`, trailing separators trimmed, lowercased. A root (`C:\`, `\`) keeps
/// its separator — trimming it would compare equal to the drive-relative `C:`.
fn dir_key(p: &Path) -> String {
    let stripped = crate::util::strip_verbatim_prefix(p);
    let folded = stripped.to_string_lossy().replace('/', "\\");
    let trimmed = folded.trim_end_matches('\\');
    if trimmed.is_empty() || trimmed.ends_with(':') {
        folded.to_lowercase()
    } else {
        trimmed.to_lowercase()
    }
}

// ── cd grammar ───────────────────────────────────────────────────────────

/// Scan a `cd` member's words (the verb excluded) under cmd.exe's grammar:
/// `/d` — its only switch, changing the drive as well as the directory — is
/// recognised case-insensitively and only immediately after the verb; a bare
/// `cd` prints the cwd and changes nothing, so it is [`CdScan::Bare`]. `cmd`
/// expands no `~` and has no `-` convention: `-`, `~` and `~/x` are ordinary
/// directory names and resolve literally. Anything else switch-shaped (`/x`)
/// is [`CdScan::BadOption`], as is a word whose quote characters cmd.exe would
/// read differently than the spelling suggests.
///
/// `words` are the member's words as cmd.exe splits them — [`tokenize`]'s raw
/// spellings, so a quoted target keeps its inner spaces and its quotes are
/// judged rather than pre-stripped by the caller.
#[must_use]
pub(super) fn cd_scan<'a>(words: &[&'a str]) -> CdScan<'a> {
    let mut i = usize::from(words.first().is_some_and(|w| w.eq_ignore_ascii_case("/d")));
    let Some(word) = words.get(i) else {
        return CdScan::Bare;
    };
    if word.starts_with('/') || word.contains('\'') {
        return CdScan::BadOption;
    }
    let Some(target) = strip_double_quotes(word) else {
        // An interior or unbalanced `"` is a spelling cmd reads differently.
        return CdScan::BadOption;
    };
    if target.is_empty() {
        return CdScan::BadOption;
    }
    i += 1;
    CdScan::Target(target, i)
}

// ── fnmatch ──────────────────────────────────────────────────────────────

/// POSIX `fnmatch` without pathname semantics, for the Windows lane where no
/// libc `fnmatch` exists: `*` (any run, `/` included), `?`, bracket expressions
/// (`[abc]`, `[!abc]`, `[^abc]`, ranges `a-z`) and `\` escaping the next
/// character. `period` implements `FNM_PERIOD`: a leading `.` in `name` is
/// matched only by a literal `.` in the pattern. A malformed bracket
/// expression (unterminated, empty, inverted range) matches nothing.
#[must_use]
pub(super) fn fnmatch(pattern: &str, name: &str, period: bool) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let n: Vec<char> = name.chars().collect();
    matches_from(&p, 0, &n, 0, period)
}

/// The bracket expression at `p[i] == '['`: whether `c` is in the class, and
/// the index just past the closing `]`. `None` for a malformed expression — an
/// unterminated or empty class, or an inverted range (fail-closed: the host
/// implementations match nothing there either).
fn bracket_match(p: &[char], i: usize, c: char) -> Option<(bool, usize)> {
    let mut j = i + 1;
    let negated = matches!(p.get(j), Some('!' | '^'));
    if negated {
        j += 1;
    }
    let mut matched = false;
    // A `]` directly after the (optional) negation is a member, not the close.
    if p.get(j) == Some(&']') {
        matched = c == ']';
        j += 1;
    }
    loop {
        let start = *p.get(j)?;
        if start == ']' {
            return Some((matched != negated, j + 1));
        }
        // `\` escapes the next character (FNM_NOESCAPE is not set).
        let (start, next) = if start == '\\' {
            (*p.get(j + 1)?, j + 2)
        } else {
            (start, j + 1)
        };
        // A range is `x-y` with the `-` neither first nor last.
        if p.get(next) == Some(&'-') && p.get(next + 1).is_some_and(|&end| end != ']') {
            let end = p[next + 1];
            if start > end {
                return None;
            }
            if (start..=end).contains(&c) {
                matched = true;
            }
            j = next + 2;
            continue;
        }
        if start == c {
            matched = true;
        }
        j = next;
    }
}

/// One step of [`fnmatch`]: compare `p` from `pi` against `n` from `ni`.
fn matches_from(p: &[char], mut pi: usize, n: &[char], mut ni: usize, period: bool) -> bool {
    // FNM_PERIOD: a leading dot is matched only by a literal dot in the
    // pattern, never by a wildcard or a class.
    if period && ni == 0 && n.first() == Some(&'.') {
        let literal_dot =
            p.get(pi) == Some(&'.') || (p.get(pi) == Some(&'\\') && p.get(pi + 1) == Some(&'.'));
        if !literal_dot {
            return false;
        }
    }
    loop {
        let Some(&c) = p.get(pi) else {
            return ni == n.len();
        };
        match c {
            '*' => {
                while p.get(pi) == Some(&'*') {
                    pi += 1;
                }
                if pi == p.len() {
                    // A trailing `*` matches any remaining run, the empty one
                    // included — the leading-dot rule was applied on entry.
                    return true;
                }
                let mut k = ni;
                loop {
                    if matches_from(p, pi, n, k, period) {
                        return true;
                    }
                    if k == n.len() {
                        return false;
                    }
                    k += 1;
                }
            }
            '?' => {
                if ni == n.len() {
                    return false;
                }
                pi += 1;
                ni += 1;
            }
            '[' => {
                if ni == n.len() {
                    return false;
                }
                let Some((matched, next)) = bracket_match(p, pi, n[ni]) else {
                    return false;
                };
                if !matched {
                    return false;
                }
                pi = next;
                ni += 1;
            }
            '\\' => {
                // A trailing backslash stands for itself (there is no character
                // left to escape), matching the host implementations.
                let (literal, step) = match p.get(pi + 1) {
                    Some(&next) => (next, 2),
                    None => ('\\', 1),
                };
                if n.get(ni) != Some(&literal) {
                    return false;
                }
                pi += step;
                ni += 1;
            }
            literal => {
                if n.get(ni) != Some(&literal) {
                    return false;
                }
                pi += 1;
                ni += 1;
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────
// The whole model is pure string/path work driven by a runtime platform value,
// so it is exercised from this host: no Windows host is needed to pin it.

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert the segmenter's `(segment, following-connector)` pairs for a line
    /// it is expected to read.
    fn assert_segments(command: &str, expected: &[(&str, &str)]) {
        let got = segment_command(command)
            .unwrap_or_else(|| panic!("{command:?}: expected a readable line"));
        let want: Vec<(String, String)> = expected
            .iter()
            .map(|(s, c)| ((*s).to_string(), (*c).to_string()))
            .collect();
        assert_eq!(got, want, "{command:?}");
    }

    #[test]
    fn connectors_split_like_cmd() {
        assert_segments("a & b", &[("a", "&"), ("b", "")]);
        assert_segments("a && b", &[("a", "&&"), ("b", "")]);
        assert_segments("a | b", &[("a", "|"), ("b", "")]);
        assert_segments("a || b", &[("a", "||"), ("b", "")]);
        assert_segments("echo a & grep x f", &[("echo a", "&"), ("grep x f", "")]);
        assert_segments("  a  &&  b  ", &[("a", "&&"), ("b", "")]);
        // A bare `&` at the end of a `&&`/`||` run is already consumed; a lone
        // trailing connector leaves cmd an empty command.
        assert!(segment_command("a &").is_none());
        assert!(segment_command("a &&").is_none());
        assert!(segment_command("a |").is_none());
        assert!(segment_command("a ||").is_none());
        assert!(segment_command("&& a").is_none());
        assert!(segment_command("| a").is_none());
    }

    #[test]
    fn quoting_and_unreadable_shapes() {
        // Inside a double-quoted span every connector and metacharacter is
        // literal; `""` inside the span is a literal quote.
        assert_segments("\"a & b\"", &[("\"a & b\"", "")]);
        assert_segments("echo \"x | y\"", &[("echo \"x | y\"", "")]);
        assert_segments("grep \"a\"\"&b\" f", &[("grep \"a\"\"&b\" f", "")]);
        // A redirect keeps its own `&`: the descriptor merge is part of the
        // member, not a separator.
        assert_segments("a 2>&1 b", &[("a 2>&1 b", "")]);
        assert_segments("a >&2 b", &[("a >&2 b", "")]);
        assert_segments("a >& out b", &[("a >& out b", "")]);
        // `;` is ordinary text: one member, never a connector.
        assert_segments("a;b", &[("a;b", "")]);
        assert_segments(
            "grep -n x a.txt; echo done",
            &[("grep -n x a.txt; echo done", "")],
        );
        // `^` and `(...)` are unmodellable outside quotes; inside they are not.
        assert!(segment_command("echo ^& grep x f").is_none());
        assert!(segment_command("(a) & b").is_none());
        assert_segments("echo \"a^b(c)\"", &[("echo \"a^b(c)\"", "")]);
        // Unbalanced quotes.
        assert!(segment_command("\"a & b").is_none());
    }

    #[test]
    fn newlines_separate_and_crlf_is_trimmed() {
        assert_segments("a\nb", &[("a", "\n"), ("b", "")]);
        assert_segments("a\r\nb\r\n", &[("a", "\n"), ("b", "\n")]);
        // Blank lines stay valid, like the unix segmenter's exception.
        assert_segments("a\n\nb", &[("a", "\n"), ("b", "")]);
        assert_segments("a\n", &[("a", "\n")]);
        // A member before a hard connector must not be empty.
        assert!(segment_command("\n& a").is_none());
    }

    #[test]
    fn unquoted_semicolon_sees_only_outside_quotes() {
        assert!(unquoted_semicolon("grep x a; b"));
        assert!(!unquoted_semicolon("grep \"a;b\" f"));
        assert!(!unquoted_semicolon("echo a\"\"b"));
        assert!(unquoted_semicolon("echo \"a;b\" & c;d"));
    }

    #[test]
    fn grep_verb_folds_case_and_the_exe_suffix() {
        let rows = [
            ("grep", Some("grep")),
            ("GREP", Some("grep")),
            ("Grep", Some("grep")),
            ("grep.exe", Some("grep")),
            ("GREP.EXE", Some("grep")),
            ("Grep.EXE", Some("grep")),
            ("Grep.Exe", Some("grep")),
            ("egrep.exe", Some("egrep")),
            ("fgrep.exe", Some("fgrep")),
            ("\"grep\"", Some("grep")),
            ("findstr", None),
            ("grep.com", None),
            ("grep.bat", None),
            (r"C:\bin\grep.exe", None),
            ("/usr/bin/grep", None),
            ("", None),
            ("\"grep", None),
        ];
        for (word, expected) in rows {
            assert_eq!(grep_verb(word), expected, "{word}");
        }
    }

    #[test]
    fn tokenize_reads_cmd_words() {
        let words = tokenize("grep \"a b\" c").expect("balanced quotes");
        let got: Vec<(&str, &str)> = words
            .iter()
            .map(|w| (w.value.as_str(), w.raw.as_str()))
            .collect();
        assert_eq!(got, [("grep", "grep"), ("a b", "\"a b\""), ("c", "c")]);

        // `""` inside a quoted span is a literal quote; an empty quoted word is
        // one (empty) argument. `a""b` is outside any span — the first quote
        // opens it and the second closes it — so cmd delivers the two letters.
        let words = tokenize("grep \"a\"\"b\" \"\"").expect("balanced quotes");
        let got: Vec<&str> = words.iter().map(|w| w.value.as_str()).collect();
        assert_eq!(got, ["grep", "a\"b", ""]);
        let words = tokenize("grep a\"\"b").expect("balanced quotes");
        assert_eq!(words[1].value, "ab");

        // `'` and `\` are ordinary characters, and `$` stays literal (cmd
        // expands neither; `$` is an ERE anchor).
        let words = tokenize("grep 'a\\b' ^$ f").expect("balanced quotes");
        let got: Vec<&str> = words.iter().map(|w| w.value.as_str()).collect();
        assert_eq!(got, ["grep", "'a\\b'", "^$", "f"]);

        // Whitespace inside quotes does not split the word.
        let words = tokenize("grep \"a|b\" f").expect("balanced quotes");
        assert_eq!(words.len(), 3);
        assert_eq!(words[1].value, "a|b");
        assert!(!words[1].redirect);

        assert!(tokenize("grep \"a").is_none());
    }

    #[test]
    fn tokenize_classifies_redirects_through_the_shared_classifier() {
        let words = tokenize("grep x f 2>&1").expect("balanced quotes");
        let last = words.last().expect("a token");
        assert!(
            last.redirect && !last.needs_target,
            "2>&1 is self-contained"
        );

        let words = tokenize("grep x f > out.txt").expect("balanced quotes");
        assert!(words[3].redirect && words[3].needs_target);

        let words = tokenize("grep x < in.txt").expect("balanced quotes");
        assert!(words[2].redirect && words[2].needs_target);
    }

    #[test]
    fn unquote_word_is_idempotent_on_token_values() {
        assert_eq!(unquote_word("\"a b\""), "a b");
        assert_eq!(unquote_word("\"a\"\"b\""), "a\"b");
        assert_eq!(unquote_word("a\"\"b"), "ab");
        assert_eq!(unquote_word("a b"), "a b");
        assert_eq!(unquote_word("'a'"), "'a'");
        assert_eq!(unquote_word(""), "");
    }

    #[test]
    fn cmd_quote_refuses_only_what_cmd_rewrites_inside_quotes() {
        assert_eq!(cmd_quote(r"C:\ws\a b.txt").unwrap(), "\"C:\\ws\\a b.txt\"");
        assert_eq!(cmd_quote("x").unwrap(), "\"x\"");
        // Metacharacters cmd stops treating as syntax inside a double-quoted
        // argument: quoting them is enough (this is what makes an install under
        // `C:\Program Files (x86)` servable).
        for safe in ["a&b", "a|b", "a<b", "a>b", "a(b", "a;b", "a^b", "a=b"] {
            assert_eq!(cmd_quote(safe).unwrap(), format!("\"{safe}\""));
        }
        // What cmd rewrites even inside quotes.
        for bad in ["%TEMP%", "a!b", "a\"b", "a\nb", "a\rb"] {
            assert!(cmd_quote(bad).is_err(), "{bad:?} must be refused");
        }
    }

    #[test]
    fn traversal_display_joins_with_the_windows_separator() {
        assert_eq!(
            traversal_display(r"C:\ws", r"src\main.rs"),
            r"C:\ws\src\main.rs"
        );
        // An operand that already ends in a separator keeps exactly that one.
        assert_eq!(
            traversal_display(r"C:\ws\", r"src\main.rs"),
            r"C:\ws\src\main.rs"
        );
        assert_eq!(traversal_display("\\", r"src\main.rs"), r"\src\main.rs");
        assert_eq!(traversal_display("/", "src"), "/src");
    }

    #[test]
    fn spec_files_are_allocated_uniquely_and_written_by_path() {
        let first = spec_file_path();
        let second = spec_file_path();
        assert_ne!(first, second, "every member gets its own file");
        assert_eq!(first.extension(), Some("json".as_ref()), "{first:?}");

        // The writer owns the directory and the file: it creates the former and
        // replaces the latter, so a leftover entry cannot turn a served member
        // into a refusal. (A `TempDir` fixture, so the host lane leaves nothing
        // of its own behind — the scratch directory itself is the daemon's.)
        let dir = tempfile::TempDir::new().expect("temp dir");
        let path = dir.path().join("nested").join("one.json");
        write_spec_file(&path, "{\"a\":1}").expect("write");
        assert_eq!(fs::read_to_string(&path).expect("read"), "{\"a\":1}");
        write_spec_file(&path, "{\"b\":2}").expect("rewrite");
        assert_eq!(fs::read_to_string(&path).expect("read"), "{\"b\":2}");

        // A directory is not writable as a file: an error the caller refuses the
        // member on, never a silent skip.
        assert!(write_spec_file(dir.path(), "{}").is_err());
    }

    #[test]
    fn is_absolute_word_reads_windows_spellings() {
        let rows = [
            (r"C:\ws", true),
            ("C:/ws", true),
            (r"\\srv\share", true),
            ("//srv/share", true),
            (r"\ws", true),
            ("/ws", true),
            ("C:ws", false),
            (r"C:", false),
            (r"sub\ws", false),
            ("ws", false),
            ("", false),
        ];
        for (word, expected) in rows {
            assert_eq!(is_absolute_word(word), expected, "{word}");
        }
    }

    #[test]
    fn glob_patterns_split_and_join_by_their_anchor() {
        let rows: &[(&str, &str, &[&str])] = &[
            (r"C:\ws\*.txt", r"C:\", &["ws", "*.txt"]),
            ("C:/ws/*", "C:/", &["ws", "*"]),
            (r"\\srv\share\a\*", r"\\srv\share\", &["a", "*"]),
            ("//srv/share/*", "//srv/share/", &["*"]),
            (r"\sub\*", r"\", &["sub", "*"]),
            ("/sub/*", "/", &["sub", "*"]),
            (r"sub\*.txt", "", &["sub", "*.txt"]),
            ("sub/*.txt", "", &["sub", "*.txt"]),
            ("*.txt", "", &["*.txt"]),
        ];
        for (pattern, prefix, comps) in rows {
            let got = split_components(pattern).unwrap_or_else(|| panic!("{pattern}"));
            assert_eq!(got.0, *prefix, "prefix of {pattern}");
            let got_c: Vec<&str> = got.1.iter().map(String::as_str).collect();
            assert_eq!(got_c, *comps, "components of {pattern}");
        }
        for bad in [r"C:foo\*", r"\\srv\*", r"\\srv\share", "", r"C:\", r"\\"] {
            assert!(split_components(bad).is_none(), "{bad} has no walk root");
        }

        // Separator runs collapse: a component is never empty.
        let (prefix, comps) = split_components(r"a\\b\*").expect("relative pattern");
        assert!(
            prefix.is_empty(),
            "a relative pattern keeps no prefix: {prefix}"
        );
        assert_eq!(comps, ["a", "b", "*"]);

        assert_eq!(join_display(r"C:\ws", "a.txt"), r"C:\ws\a.txt");
        assert_eq!(join_display(r"C:\", "ws"), r"C:\ws");
        assert_eq!(join_display("", "ws"), "ws");
        assert_eq!(join_display("/", "ws"), "/ws");
        // Round trip: every component joined back onto its prefix is a path
        // spelled the way the operand was typed.
        for (pattern, prefix, comps) in rows {
            let mut built = (*prefix).to_string();
            for c in *comps {
                built = join_display(&built, c);
            }
            assert_eq!(
                built.replace('/', "\\"),
                pattern.replace('/', "\\"),
                "round trip of {pattern}"
            );
        }
    }

    #[test]
    fn has_unquoted_glob_respects_quoting() {
        assert!(has_unquoted_glob("*.txt"));
        assert!(has_unquoted_glob("a?c"));
        assert!(has_unquoted_glob("[ab]c"));
        assert!(!has_unquoted_glob("\"*.txt\""));
        assert!(!has_unquoted_glob("plain.txt"));
        assert!(!has_unquoted_glob("\"a\"\"*\""));
        assert!(has_unquoted_glob("\"a\"\"b\"*"));
    }

    #[test]
    fn same_directory_ignores_case_separators_and_the_verbatim_prefix() {
        assert!(same_directory(Path::new(r"C:\ws"), Path::new(r"C:\WS\")));
        assert!(same_directory(Path::new(r"C:\ws"), Path::new(r"\\?\C:\ws")));
        assert!(same_directory(Path::new("C:/ws/"), Path::new(r"C:\ws")));
        assert!(same_directory(
            Path::new(r"\\?\UNC\srv\share"),
            Path::new(r"\\SRV\Share")
        ));
        assert!(!same_directory(Path::new(r"C:\ws\a"), Path::new(r"C:\ws")));
        // A drive root is not its drive-relative spelling.
        assert!(!same_directory(Path::new(r"C:\"), Path::new("C:")));
        assert!(same_directory(Path::new(r"C:\"), Path::new("C:/")));
    }

    #[test]
    fn cd_grammar_is_cmd_shaped() {
        assert_eq!(cd_scan(&[]), CdScan::Bare);
        assert_eq!(cd_scan(&["/d"]), CdScan::Bare);
        assert_eq!(cd_scan(&["sub"]), CdScan::Target("sub", 1));
        assert_eq!(cd_scan(&["/D", "sub"]), CdScan::Target("sub", 2));
        // `~`, `-` and `~/x` are ordinary names: cmd expands none of them.
        assert_eq!(cd_scan(&["~"]), CdScan::Target("~", 1));
        assert_eq!(cd_scan(&["-"]), CdScan::Target("-", 1));
        assert_eq!(cd_scan(&["~", "sub"]), CdScan::Target("~", 1));
        assert_eq!(cd_scan(&[r"~\sub"]), CdScan::Target(r"~\sub", 1));
        assert_eq!(cd_scan(&[r"C:\ws"]), CdScan::Target(r"C:\ws", 1));
        assert_eq!(cd_scan(&["\"my dir\""]), CdScan::Target("my dir", 1));
        // Switch-shaped words and quote spellings cmd reads differently.
        assert_eq!(cd_scan(&["/x"]), CdScan::BadOption);
        assert_eq!(cd_scan(&["-P"]), CdScan::Target("-P", 1));
        assert_eq!(cd_scan(&["'sub'"]), CdScan::BadOption);
        assert_eq!(cd_scan(&["\"sub"]), CdScan::BadOption);
        assert_eq!(cd_scan(&["\"\""]), CdScan::BadOption);
        assert_eq!(cd_scan(&["/d", "/d"]), CdScan::BadOption);
    }
}
