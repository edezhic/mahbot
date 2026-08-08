//! Transparent grep/egrep/fgrep interception for the read-only shell tool.
//!
//! The read-only shell guard passes commands to a real shell. Grep-family
//! invocations that can be served with byte-identical behavior are rewritten
//! to a hidden `__grep-engine` subcommand of the current binary (dispatched
//! before instance-lock acquisition in `main()`), which runs the ripgrep
//! substrate (grep-regex/grep-searcher + the ignore crate for rg-default
//! recursive-walk exclusions). Anything not provably safe executes the
//! original command unchanged (fallback); the engine itself re-validates and
//! `exec`s the real grep on any runtime doubt.
//!
//! Parity target: the host system grep (BSD grep on macOS) under the shell
//! tool's pinned `LC_ALL=C.UTF-8` environment. The one approved behavioral
//! delta: recursive walks skip hidden/gitignored content (rg defaults).
//! The differential matrix (macOS-gated) is the authoritative parity gate.

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use grep_matcher::Matcher;
use serde::{Deserialize, Serialize};

use crate::tools::shell::SHELL_PIPE_READ_CAP;
use crate::tools::shell::readonly::{strip_heredoc_bodies, strip_outer_quotes};

// ── Protocol constants ────────────────────────────────────────────────────

/// Hidden subcommand name dispatched before lock acquisition (like `debug`).
const ENGINE_VERB: &str = "__grep-engine";
/// Spec JSON protocol version; mismatches make the engine fall back.
/// Bumped for the -c/-l GrepFlags fields: a spec from a swapped binary is
/// caught here (both directions) and execs the real grep via the fallback
/// argv, which carries the new flags.
const PROTOCOL_VERSION: u32 = 3;
/// NUL-detection window for binary files (FreeBSD grep reads 32 KiB).
const BINARY_WINDOW: usize = 32 * 1024;
/// Engine self-cap on written output; the shell pipe reader caps at the same.
const OUTPUT_CAP: usize = SHELL_PIPE_READ_CAP;
/// Sentinel exit code: engine could not serve and could not exec grep either.
/// The parent re-runs the original command on this code.
pub(super) const ENGINE_FAILED_EXIT: i32 = 3;
/// Stderr signature of a stale self-update binary (one lacking the hidden
/// subcommand) running full main() and dying at instance-lock acquisition —
/// its exit 1 is a legitimate grep no-match code, so the parent re-runs on
/// this message instead. Mirrors `self_update::acquire_lock`'s error text.
pub(super) const STALE_BINARY_LOCK_MSG: &str = "Another instance of mahbot is already running";
/// Specs larger than this fall back (argv-size hygiene).
const MAX_SPEC_JSON: usize = 64 * 1024;
/// Searcher line-buffer cap; exceeding it is a grep-style error (exit 2).
const HEAP_LIMIT: usize = 512 * 1024 * 1024;

// ── Spec: the JSON protocol between parent and engine ─────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(non_snake_case, clippy::struct_excessive_bools)] // flag-letter names, grep CLI surface
struct EngineSpec {
    /// Protocol version (mismatch → engine falls back).
    version: u32,
    /// Original verb ("grep"/"egrep"/"fgrep") — error-message prefix.
    verb: String,
    /// Matching mode (BRE/ERE/fixed).
    mode: MatchMode,
    /// Parsed flag surface (single source for parent and engine).
    flags: GrepFlags,
    /// Ordered --include/--exclude filters; last match wins (BSD semantics).
    filters: Vec<(bool, String)>,
    /// --exclude-dir patterns (any match excludes a traversed dir).
    exclude_dir: Vec<String>,
    /// Engine-dialect patterns (BRE translated, ERE/Fixed as-is).
    patterns: Vec<String>,
    operands: Vec<Operand>,
    /// Expected canonical working directory (the parent's tracked cwd).
    cwd: String,
    /// Original post-expansion argv for the exec-in-place fallback.
    fallback: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Operand {
    /// Path as grep would print it (post-expansion, as traversed).
    display: String,
    /// Absolute path for opening.
    resolved: String,
    /// Operand spelling ended with `/` (dereference rule for symlinks).
    trailing_slash: bool,
}

// ── Parent side: command analysis ─────────────────────────────────────────

/// Try to rewrite `command` into an engine-served equivalent; `None` = the
/// original command executes unchanged. Read-only mode only (checked by the
/// caller); Unix only.
pub(super) fn try_serve_command(command: &str, workspace_root: &Path) -> Option<String> {
    let home = pinned_home()?;
    let (specs, rewritten) = match analyze_command(command, workspace_root, &home, false) {
        Ok(v) => v,
        Err(reason) => {
            tracing::debug!(command = command, %reason, "grep engine: fallback");
            return None;
        }
    };
    if !engine_available() {
        return None;
    }
    for spec in &specs {
        if !spec_json_ok(spec) {
            return None;
        }
    }
    tracing::debug!(
        command = command,
        greps = specs.len(),
        "grep engine: served"
    );
    Some(rewritten)
}

/// Rewrite with an explicit home (subprocess harness uses a fixture home for
/// `~` operands); production gate (single-file lookups stay on real grep).
#[cfg(feature = "grep-engine-e2e")]
#[doc(hidden)]
#[must_use]
pub fn try_serve_command_for_test(
    command: &str,
    workspace_root: &Path,
    home: &Path,
) -> Option<String> {
    let (specs, rewritten) = analyze_command(command, workspace_root, home, false).ok()?;
    for spec in &specs {
        if !spec_json_ok(spec) {
            return None;
        }
    }
    Some(rewritten)
}

/// The shell tool's pinned `$HOME` — the child shell's synthetic baseline
/// (`UserDirs`), never the ambient daemon env.
fn pinned_home() -> Option<PathBuf> {
    directories::UserDirs::new().map(|d| d.home_dir().to_path_buf())
}

/// Spec JSON must stay within the argv-size hygiene bound.
fn spec_json_ok(spec: &EngineSpec) -> bool {
    serde_json::to_string(spec).is_ok_and(|j| j.len() <= MAX_SPEC_JSON)
}

/// Per-process probe result (cached — a probe spawns the full binary, so it
/// must not run per call). Real stale-binary safety nets: the engine's version
/// check + exec-in-place fallback, and the parent's lock-message re-run for
/// binaries lacking the subcommand entirely (a probe-to-dispatch swap remains
/// possible and is covered by those).
static ENGINE_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn engine_available() -> bool {
    *ENGINE_AVAILABLE.get_or_init(|| {
        let Some(exe) = std::env::current_exe().ok() else {
            return false;
        };
        std::process::Command::new(exe)
            .arg(ENGINE_VERB)
            .arg("--probe")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    })
}

/// Rewrite a spec into the shell command fragment the engine runs as.
fn build_rewritten(spec: &EngineSpec) -> String {
    let json = serde_json::to_string(spec).expect("spec serializes");
    let exe = std::env::current_exe().expect("current exe resolved");
    format!(
        "{} {} {}",
        shell_quote(&exe.to_string_lossy()),
        ENGINE_VERB,
        shell_quote(&json)
    )
}

/// Lossless single-quote shell quoting (embedded `'` → `'\''`).
fn shell_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        if c == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}

/// Why a command was not served (telemetry + fail-closed decisions).
#[derive(Debug)]
enum Fallback {
    NoGrep,
    NestedGrep,
    Heredoc,
    StdinMode,
    UnsupportedFlag(String),
    MissingOptionValue,
    EmptyAlternation,
    Pattern(String),
    CompileFailure(String),
    UnresolvableOperand(String),
    UnexpandableGlob,
    SingleFile,
    CdUntrackable,
    PipelineShape,
    SegmentEmpty,
}

impl std::fmt::Display for Fallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fallback::NoGrep => write!(f, "no grep"),
            Fallback::NestedGrep => write!(f, "nested grep"),
            Fallback::Heredoc => write!(f, "heredoc present"),
            Fallback::StdinMode => write!(f, "stdin mode"),
            Fallback::UnsupportedFlag(s) => write!(f, "unsupported flag {s}"),
            Fallback::MissingOptionValue => write!(f, "missing option value"),
            Fallback::EmptyAlternation => write!(f, "empty alternation"),
            Fallback::Pattern(s) => write!(f, "pattern: {s}"),
            Fallback::CompileFailure(s) => write!(f, "compile failure: {s}"),
            Fallback::UnresolvableOperand(s) => write!(f, "unresolvable operand: {s}"),
            Fallback::UnexpandableGlob => write!(f, "unexpandable glob"),
            Fallback::SingleFile => write!(f, "single file"),
            Fallback::CdUntrackable => write!(f, "cd untrackable"),
            Fallback::PipelineShape => write!(f, "pipeline shape"),
            Fallback::SegmentEmpty => write!(f, "empty command"),
        }
    }
}

/// Analyze a full shell command: segment it, track cds, serve every grep
/// member, keep everything else verbatim. Returns the rewritten command plus
/// one spec per served grep (in command order) or the first fallback reason.
fn analyze_command(
    command: &str,
    workspace_root: &Path,
    home: &Path,
    allow_single: bool,
) -> Result<(Vec<EngineSpec>, String), Fallback> {
    let stripped = strip_heredoc_bodies(command);
    if stripped != command {
        // strip_heredoc_bodies drops the `<<` marker, body and terminator (they
        // are stripped for read-only scanning). Rewriting around them would
        // leave a bare non-grep member reading inherited stdin — fail-closed.
        return Err(Fallback::Heredoc);
    }
    let segments = split_segments(&stripped);
    if segments.is_empty() {
        return Err(Fallback::SegmentEmpty);
    }

    // Pipeline grouping: consecutive segments joined by `|`/`|&`.
    let n = segments.len();
    let mut pstart = vec![0usize; n];
    let mut pend = vec![0usize; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && matches!(segments[j].1.as_str(), "|" | "|&") {
            j += 1;
        }
        for k in i..=j {
            pstart[k] = i;
            pend[k] = j;
        }
        i = j + 1;
    }

    let mut cwd = canonical_or_lexical(workspace_root);
    let mut rewritten: Vec<(String, String)> = Vec::new();
    let mut specs: Vec<EngineSpec> = Vec::new();
    // A served `grep | head` pipeline preserves its head member verbatim.
    let mut head_preserved = vec![false; n];

    for (idx, (seg, conn)) in segments.iter().enumerate() {
        let verb = first_word(seg);
        if is_cd_segment(verb) {
            if pstart[idx] != pend[idx] {
                return Err(Fallback::CdUntrackable);
            }
            cwd = resolve_cd(seg, &cwd, home)?;
            rewritten.push((seg.clone(), conn.clone()));
            continue;
        }
        if is_grep_verb(verb) {
            if pstart[idx] != idx {
                // Only the first member of a pipeline can be served.
                return Err(Fallback::PipelineShape);
            }
            if pend[idx] != idx {
                // In a pipeline: servable only as `grep … | head`.
                if pend[idx] != idx + 1 || !is_head_segment(&segments[pend[idx]].0) {
                    return Err(Fallback::PipelineShape);
                }
            }
            match serve_one_grep(seg, verb, &cwd, home, allow_single) {
                Ok((spec, rewritten_seg)) => {
                    if pstart[idx] != pend[idx] {
                        head_preserved[pend[idx]] = true;
                    }
                    rewritten.push((rewritten_seg, conn.clone()));
                    specs.push(spec);
                }
                Err(e) => return Err(e),
            }
            continue;
        }
        if is_head_segment(seg) && head_preserved[idx] {
            rewritten.push((seg.clone(), conn.clone()));
            continue;
        }
        // Non-grep member: verbatim, but indirect/compound grep invocations
        // make the whole command fall back (fail-closed).
        if is_compound_segment(seg) || segment_contains_grep(seg, verb) {
            return Err(Fallback::NestedGrep);
        }
        if pstart[idx] != pend[idx] {
            return Err(Fallback::PipelineShape);
        }
        rewritten.push((seg.clone(), conn.clone()));
    }

    if specs.is_empty() {
        return Err(Fallback::NoGrep);
    }
    Ok((specs, join_rewritten(&rewritten)))
}

/// Join rewritten segments with their original connectors into one command.
fn join_rewritten(segments: &[(String, String)]) -> String {
    let mut out = String::new();
    for (i, (seg, conn)) in segments.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(seg);
        if i + 1 < segments.len() {
            out.push(' ');
            out.push_str(conn);
        }
    }
    out
}

/// Split a command into (segment, following-connector) pairs. Connectors are
/// `&&`, `||`, `;`, `|`, `|&`, `\n`, or `` (last). Quote- and
/// substitution-aware, heredoc bodies already stripped.
#[allow(clippy::too_many_lines)] // quote/substitution state machine
fn split_segments(command: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut chars = command.chars().peekable();

    let flush = |current: &mut String, out: &mut Vec<(String, String)>, conn: &str| {
        let t = current.trim();
        if !t.is_empty() {
            out.push((t.to_string(), conn.to_string()));
        }
        current.clear();
    };

    while let Some(c) = chars.next() {
        if c == '\\' && !in_single {
            match chars.next() {
                Some('\n') => continue,
                Some(next) => {
                    current.push('\\');
                    current.push(next);
                }
                None => current.push('\\'),
            }
            continue;
        }
        if super::check_outside_quotes(c, &mut in_single, &mut in_double) {
            if super::consume_substitution(c, &mut chars, &mut current) {
                continue;
            }
            match c {
                '&' if chars.peek() == Some(&'&') => {
                    chars.next();
                    flush(&mut current, &mut out, "&&");
                    continue;
                }
                '|' if current.trim_end().ends_with('>') => {
                    current.push(c);
                }
                '|' => {
                    if chars.peek() == Some(&'&') {
                        chars.next();
                        flush(&mut current, &mut out, "|&");
                    } else if chars.peek() == Some(&'|') {
                        chars.next();
                        flush(&mut current, &mut out, "||");
                    } else {
                        flush(&mut current, &mut out, "|");
                    }
                    continue;
                }
                '\n' => {
                    flush(&mut current, &mut out, "\n");
                    continue;
                }
                ';' => {
                    flush(&mut current, &mut out, ";");
                    continue;
                }
                _ => {}
            }
        }
        current.push(c);
    }
    flush(&mut current, &mut out, "");
    out
}

/// First whitespace-delimited word of a segment (raw, quote-preserving).
fn first_word(segment: &str) -> &str {
    segment.split_whitespace().next().unwrap_or("")
}

fn is_grep_verb(verb: &str) -> bool {
    matches!(verb, "grep" | "egrep" | "fgrep")
}

fn is_cd_segment(verb: &str) -> bool {
    verb == "cd"
}

fn is_head_segment(segment: &str) -> bool {
    first_word(segment) == "head"
}

/// Command-introducer verbs whose argument list may contain a nested grep
/// invocation (compounds, indirect invocations). Presence of a grep-family
/// word in such a segment falls the whole command back.
const GREP_INTRODUCERS: &[&str] = &[
    "if", "while", "until", "case", "for", "select", "then", "else", "elif", "do", "!", "time",
    "command", "builtin", "exec", "eval", "sudo", "env", "nice", "nohup", "xargs", "ssh", "sh",
    "bash", "zsh", "ksh", "dash", "csh", "tcsh", "fish", "find", "git", "docker", "kubectl",
    "podman",
];

/// Compound-construct keywords: a segment starting with one (or with a `(`/`{`
/// group opener) means the command nests a compound construct, so no grep in
/// it — even in a later segment — may be served (criterion 3: compounds fall
/// back). Without this, `case … in a) grep x a;; esac` and `(cd d && grep x a
/// b)` serve the grep member, and a trailing `)` glued to an operand changes
/// the served argv.
const COMPOUND_KEYWORDS: &[&str] = &[
    "if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done", "case", "esac",
    "select",
];

/// True when a segment opens a compound construct (keyword prefix or a `(`/`{`
/// group opener) — such a command must not serve any grep.
fn is_compound_segment(segment: &str) -> bool {
    let trimmed = segment.trim_start();
    match trimmed.chars().next() {
        Some('(' | '{') => true,
        Some(c) if c.is_alphanumeric() || c == '_' => {
            COMPOUND_KEYWORDS.contains(&first_word(trimmed))
        }
        _ => false,
    }
}

/// True when a non-grep segment could contain a grep invocation we would not
/// serve (compound constructs, indirect invocation). Conservative: any
/// grep-family word in an introducer segment, or an env-assignment verb.
fn segment_contains_grep(segment: &str, verb: &str) -> bool {
    let suspicious = GREP_INTRODUCERS.contains(&verb) || verb.contains('=');
    if !suspicious {
        return false;
    }
    segment.split_whitespace().any(|w| {
        let bare = strip_outer_quotes(w).map_or(w, |(c, _)| c);
        is_grep_verb(bare)
    })
}

/// Resolve a literal `cd` segment against the tracked cwd. Returns the new
/// canonical cwd. Only statically-resolvable targets track; everything else
/// falls back (fail-closed).
fn resolve_cd(segment: &str, cwd: &Path, home: &Path) -> Result<PathBuf, Fallback> {
    let words: Vec<&str> = segment.split_whitespace().collect();
    let mut i = 1;
    let mut options_ended = false;
    let target = loop {
        let Some(w) = words.get(i) else { break None };
        if !options_ended && w.starts_with('-') && *w != "-" {
            if *w == "--" {
                options_ended = true;
            } else if !w[1..].bytes().all(|b| matches!(b, b'P' | b'L')) {
                return Err(Fallback::CdUntrackable);
            }
            i += 1;
            continue;
        }
        break Some(strip_outer_quotes(w).map_or(*w, |(c, _)| c));
    };
    let Some(target) = target else {
        // Bare `cd` → $HOME.
        return Ok(canonical_or_lexical(home));
    };
    if words.get(i + 1).is_some() {
        // Extra operands are shell-dependent — fail-closed.
        return Err(Fallback::CdUntrackable);
    }
    if has_expansion(target) {
        return Err(Fallback::CdUntrackable);
    }
    let resolved = if target == "-" {
        return Err(Fallback::CdUntrackable);
    } else if let Some(rest) = target.strip_prefix("~/") {
        home.join(rest)
    } else if target == "~" {
        home.to_path_buf()
    } else if target.starts_with('/') {
        PathBuf::from(target)
    } else {
        cwd.join(target)
    };
    Ok(canonical_or_lexical(&resolved))
}

/// Canonicalize an existing path; lexically normalize otherwise. All inputs
/// are already absolute, so `std::path::absolute` (purely lexical) is the
/// exact fallback for `fs::canonicalize`.
fn canonical_or_lexical(p: &Path) -> PathBuf {
    fs::canonicalize(p)
        .or_else(|_| std::path::absolute(p))
        .unwrap_or_else(|_| p.to_path_buf())
}

/// True when a raw shell word contains an unquoted expansion (`$`, backtick,
/// `$(`, `$'…'`) that cannot be resolved statically.
fn has_expansion(word: &str) -> bool {
    let mut in_single = false;
    let mut escaped = false;
    for c in word.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' {
            escaped = true;
            continue;
        }
        if c == '\'' {
            in_single = !in_single;
            continue;
        }
        if !in_single && (c == '$' || c == '`') {
            return true;
        }
    }
    false
}

// ── Grep invocation parsing (BSD-getopt compatible) ───────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
enum MatchMode {
    Basic,
    Extended,
    Fixed,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
// Lenient parse so a swapped binary's spec reaches the version check and
// execs the real grep in place (the sentinel would need a parent re-run).
#[serde(default)]
#[allow(non_snake_case, clippy::struct_excessive_bools)] // flag-letter names, grep CLI surface
struct GrepFlags {
    n: bool,
    i: bool,
    v: bool,
    w: bool,
    x: bool,
    a: bool,
    h: bool,
    H: bool,
    s: bool,
    r: bool,
    o: bool,
    null: bool,
    c: bool,
    l: bool,
    m: Option<u64>,
    before: usize,
    after: usize,
}

impl GrepFlags {
    /// -c/-l select lines by count/name instead of printing them; -o becomes
    /// inert and context flags are accepted but ignored, so the flag-surface
    /// interaction gates don't apply.
    fn count_mode(&self) -> bool {
        self.c || self.l
    }
}

/// A word from the grep segment: its unquoted value plus redirect metadata.
struct GrepWord {
    /// Unquoted value (expansions already rejected).
    value: String,
    /// Raw spelling for redirects preserved verbatim in the rewrite.
    raw: String,
    redirect: bool,
    needs_target: bool,
}

/// Tokenize a grep segment (after the verb): quote-aware split, unquote,
/// redirect classification. Expansions in any position fall back.
fn grep_tokenize(segment: &str) -> Result<Vec<GrepWord>, Fallback> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut chars = segment.chars().peekable();

    let flush = |current: &mut String, out: &mut Vec<GrepWord>| -> Result<(), Fallback> {
        if current.is_empty() {
            return Ok(());
        }
        let raw = std::mem::take(current);
        let value = unquote_word(&raw)?;
        let (redirect, needs_target) = classify_redirect(&raw);
        out.push(GrepWord {
            value,
            raw,
            redirect,
            needs_target,
        });
        Ok(())
    };

    while let Some(c) = chars.next() {
        if c == '\\' && !in_single {
            match chars.next() {
                Some('\n') => continue,
                Some(next) => {
                    current.push('\\');
                    current.push(next);
                }
                None => current.push('\\'),
            }
            continue;
        }
        if super::check_outside_quotes(c, &mut in_single, &mut in_double) {
            if super::consume_substitution(c, &mut chars, &mut current) {
                continue;
            }
            if c.is_whitespace() {
                flush(&mut current, &mut out)?;
                continue;
            }
        }
        current.push(c);
    }
    flush(&mut current, &mut out)?;
    Ok(out)
}

/// Unquote a raw shell word into its literal value; any expansion (unquoted
/// or double-quoted `$`/backtick, ANSI-C `$'…'`) is a fallback.
fn unquote_word(raw: &str) -> Result<String, Fallback> {
    if has_expansion(raw) {
        return Err(Fallback::UnresolvableOperand(raw.to_string()));
    }
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\'' => {
                for c2 in chars.by_ref() {
                    if c2 == '\'' {
                        break;
                    }
                    out.push(c2);
                }
            }
            '"' => {
                for c2 in chars.by_ref() {
                    if c2 == '"' {
                        break;
                    }
                    out.push(c2);
                }
            }
            '\\' => {
                if let Some(next) = chars.next() {
                    out.push(next);
                } else {
                    out.push('\\');
                }
            }
            other => out.push(other),
        }
    }
    Ok(out)
}

/// Redirect-operator classification on the raw token (quoted redirects are
/// literals). Mirrors the read-only guard's token classifier.
fn classify_redirect(w: &str) -> (bool, bool) {
    match w {
        ">" | ">&" | ">>" | ">|" | "<" | "<&" | "<>" | "&>" | "&>>" => return (true, true),
        "2>&1" | "1>&2" => return (true, false),
        _ => {}
    }
    if w.starts_with("<<<")
        || (w.len() > 3 && w.as_bytes()[0].is_ascii_digit() && w.ends_with("<<<"))
    {
        return (true, true);
    }
    if is_digit_suffix_redirect(w, b'>') || is_digit_suffix_redirect(w, b'<') {
        return (true, true);
    }
    if w.starts_with('>') || w.starts_with('<') {
        return (true, false);
    }
    if w.len() > 1 && w.as_bytes()[0].is_ascii_digit() && (w.contains('>') || w.contains('<')) {
        return (true, false);
    }
    if w.contains("&>") {
        return (true, false);
    }
    (false, false)
}

fn is_digit_suffix_redirect(w: &str, op: u8) -> bool {
    w.len() > 1
        && w.as_bytes()[0].is_ascii_digit()
        && w.as_bytes()[1] == op
        && w[2..].bytes().all(|b| b.is_ascii_digit())
}

/// A parsed, validated grep invocation ready for operand resolution.
struct ParsedGrep {
    mode: MatchMode,
    flags: GrepFlags,
    /// Ordered --include/--exclude filters (bool = include); last match wins.
    filters: Vec<(bool, String)>,
    exclude_dir: Vec<String>,
    /// Translated engine-dialect patterns.
    engine_patterns: Vec<String>,
    /// Unquoted operand tokens (pre-resolution).
    operand_tokens: Vec<String>,
    /// Redirect tokens preserved verbatim in the rewritten segment.
    redirects: Vec<String>,
    /// `verb` + normalized flags + patterns (operands appended later).
    fallback_prefix: Vec<String>,
}

#[allow(clippy::too_many_lines)] // BSD-getopt flag table is inherently long
/// Parse a grep segment's words (after the verb) with BSD-getopt semantics:
/// GNU-style permutation, value-taking options consuming the rest of their
/// token, last-wins -G/-E/-F, the macOS option cluster, and the verified flag
/// surface only — anything else falls back.
fn parse_grep_words(words: &[GrepWord], verb: &str) -> Result<ParsedGrep, Fallback> {
    // Redirects (and their targets) are removed by the shell before grep sees
    // the argv; collect them verbatim for the rewrite.
    let mut argv: Vec<String> = Vec::new();
    let mut redirects: Vec<String> = Vec::new();
    let mut i = 0;
    while i < words.len() {
        if words[i].redirect {
            redirects.push(words[i].raw.clone());
            if words[i].needs_target && i + 1 < words.len() {
                redirects.push(words[i + 1].raw.clone());
                i += 1;
            }
        } else {
            argv.push(words[i].value.clone());
        }
        i += 1;
    }

    let mut mode = match verb {
        "egrep" => MatchMode::Extended,
        "fgrep" => MatchMode::Fixed,
        _ => MatchMode::Basic,
    };
    let mut flags = GrepFlags::default();
    let mut filters = Vec::new();
    let mut exclude_dir = Vec::new();
    let mut e_patterns: Vec<String> = Vec::new();
    let mut positional: Option<String> = None;
    let mut operand_tokens: Vec<String> = Vec::new();
    let mut options_ended = false;

    let mut i = 0;
    while i < argv.len() {
        let tok = &argv[i];
        if !options_ended && tok == "--" {
            options_ended = true;
            i += 1;
            continue;
        }
        if !options_ended && tok.starts_with('-') && tok.len() > 1 {
            if let Some(rest) = tok.strip_prefix("--") {
                // Long options.
                if rest == "null" {
                    flags.null = true;
                } else if rest == "count" {
                    flags.c = true;
                } else if rest == "files-with-matches" {
                    flags.l = true;
                } else if let Some(v) = rest.strip_prefix("include=") {
                    filters.push((true, v.to_string()));
                } else if let Some(v) = rest.strip_prefix("exclude=") {
                    filters.push((false, v.to_string()));
                } else if let Some(v) = rest.strip_prefix("exclude-dir=") {
                    exclude_dir.push(v.to_string());
                } else if rest == "include" || rest == "exclude" || rest == "exclude-dir" {
                    i += 1;
                    let v = argv.get(i).ok_or(Fallback::MissingOptionValue)?;
                    match rest {
                        "include" => filters.push((true, v.clone())),
                        "exclude" => filters.push((false, v.clone())),
                        _ => exclude_dir.push(v.clone()),
                    }
                } else {
                    return Err(Fallback::UnsupportedFlag(format!("--{rest}")));
                }
            } else {
                // Short option cluster; value-taking options consume the rest.
                let chars: Vec<char> = tok[1..].chars().collect();
                let mut j = 0;
                while j < chars.len() {
                    match chars[j] {
                        'n' => flags.n = true,
                        'i' | 'y' => flags.i = true,
                        'v' => flags.v = true,
                        'w' => flags.w = true,
                        'x' => flags.x = true,
                        'a' => flags.a = true,
                        'h' => flags.h = true,
                        'H' => flags.H = true,
                        's' => flags.s = true,
                        'r' | 'R' => flags.r = true,
                        'o' => flags.o = true,
                        'c' => flags.c = true,
                        'l' => flags.l = true,
                        'u' => {} // accepted no-op
                        'G' => mode = MatchMode::Basic,
                        'E' => mode = MatchMode::Extended,
                        'F' => mode = MatchMode::Fixed,
                        'e' | 'm' | 'A' | 'B' | 'C' => {
                            let rest: String = chars[j + 1..].iter().collect();
                            let value = if rest.is_empty() {
                                i += 1;
                                argv.get(i).cloned().ok_or(Fallback::MissingOptionValue)?
                            } else {
                                rest
                            };
                            match chars[j] {
                                'e' => e_patterns.push(value),
                                'm' => {
                                    let n: u64 = value
                                        .parse()
                                        .map_err(|_| Fallback::UnsupportedFlag("-m".into()))?;
                                    if n == 0 {
                                        return Err(Fallback::UnsupportedFlag("-m0".into()));
                                    }
                                    flags.m = Some(n);
                                }
                                'A' => {
                                    flags.after = value
                                        .parse()
                                        .map_err(|_| Fallback::UnsupportedFlag("-A".into()))?;
                                }
                                'B' => {
                                    flags.before = value
                                        .parse()
                                        .map_err(|_| Fallback::UnsupportedFlag("-B".into()))?;
                                }
                                'C' => {
                                    let n: usize = value
                                        .parse()
                                        .map_err(|_| Fallback::UnsupportedFlag("-C".into()))?;
                                    flags.before = n;
                                    flags.after = n;
                                }
                                _ => unreachable!(),
                            }
                            break; // rest of the cluster was the value
                        }
                        other => {
                            return Err(Fallback::UnsupportedFlag(format!("-{other}")));
                        }
                    }
                    j += 1;
                }
            }
            i += 1;
        } else if positional.is_none() && e_patterns.is_empty() {
            positional = Some(tok.clone());
            i += 1;
        } else {
            operand_tokens.push(tok.clone());
            i += 1;
        }
    }

    // Flag-surface validation (criterion 9). Under -c/-l, -o is inert and
    // context flags are accepted but ignored, so their interaction gates
    // don't apply.
    if flags.v {
        flags.o = false; // -v -o behaves as plain -v
    }
    let count_mode = flags.count_mode();
    if flags.v && (flags.before > 0 || flags.after > 0) && !count_mode {
        return Err(Fallback::UnsupportedFlag("-v+context".into()));
    }
    if flags.o && (flags.before > 0 || flags.after > 0) && !count_mode {
        return Err(Fallback::UnsupportedFlag("-o+context".into()));
    }
    if flags.o && flags.m.is_some() && !count_mode {
        return Err(Fallback::UnsupportedFlag("-m+-o".into()));
    }

    // Patterns: BSD splits each on newlines (add_pattern); an empty piece
    // among several is an empty-alternation fallback; a single empty pattern
    // matches everything and is servable.
    let mut raw_patterns: Vec<String> = Vec::new();
    for p in positional.into_iter().chain(e_patterns) {
        raw_patterns.extend(p.split('\n').map(str::to_string));
    }
    if raw_patterns.is_empty() {
        return Err(Fallback::Pattern("no pattern".into()));
    }
    if raw_patterns.len() > 1 && raw_patterns.iter().any(String::is_empty) {
        return Err(Fallback::EmptyAlternation);
    }

    // Translation + compile validation happen here (parent side) so most
    // runtime fallbacks are eliminated before dispatch.
    let mut engine_patterns = Vec::with_capacity(raw_patterns.len());
    for p in &raw_patterns {
        engine_patterns.push(translate_pattern(p, mode)?);
    }
    if flags.w {
        for p in &engine_patterns {
            if !word_safe(p) {
                return Err(Fallback::Pattern("-w edge".into()));
            }
        }
    }
    // `-o` + alternation: the engine (leftmost-first) diverges from BSD
    // (leftmost-longest) on match length — fail-closed. Fixed-string mode has
    // no alternation operator, so a literal `|` stays servable. -o is inert
    // under -c/-l, so the divergence cannot surface there.
    if flags.o
        && !count_mode
        && mode != MatchMode::Fixed
        && engine_patterns.iter().any(|p| has_alternation(p))
    {
        return Err(Fallback::UnsupportedFlag("-o+alternation".into()));
    }
    let matcher = build_matcher(&engine_patterns, mode, &flags)
        .map_err(|e| Fallback::CompileFailure(e.to_string()))?;
    // Empty-matchable patterns diverge between BSD and the engine; the single
    // literal-empty pattern is the one approved exception (matches everything).
    let matches_empty = matcher
        .is_match(b"")
        .map_err(|e| Fallback::CompileFailure(e.to_string()))?;
    if matches_empty && !(raw_patterns.len() == 1 && raw_patterns[0].is_empty()) {
        return Err(Fallback::EmptyAlternation);
    }
    drop(matcher);

    let mut fallback_prefix = vec![verb.to_string()];
    match (verb, mode) {
        ("grep" | "fgrep", MatchMode::Extended) => fallback_prefix.push("-E".into()),
        ("grep" | "egrep", MatchMode::Fixed) => fallback_prefix.push("-F".into()),
        ("egrep" | "fgrep", MatchMode::Basic) => fallback_prefix.push("-G".into()),
        _ => {}
    }
    push_flag(&mut fallback_prefix, flags.n, "-n");
    push_flag(&mut fallback_prefix, flags.i, "-i");
    push_flag(&mut fallback_prefix, flags.v, "-v");
    push_flag(&mut fallback_prefix, flags.w, "-w");
    push_flag(&mut fallback_prefix, flags.x, "-x");
    push_flag(&mut fallback_prefix, flags.a, "-a");
    push_flag(&mut fallback_prefix, flags.h, "-h");
    push_flag(&mut fallback_prefix, flags.H, "-H");
    push_flag(&mut fallback_prefix, flags.s, "-s");
    push_flag(&mut fallback_prefix, flags.r, "-r");
    push_flag(&mut fallback_prefix, flags.o, "-o");
    push_flag(&mut fallback_prefix, flags.c, "-c");
    push_flag(&mut fallback_prefix, flags.l, "-l");
    if flags.null {
        fallback_prefix.push("--null".into());
    }
    if let Some(m) = flags.m {
        fallback_prefix.push(format!("-m{m}"));
    }
    if flags.before > 0 {
        fallback_prefix.push(format!("-B{}", flags.before));
    }
    if flags.after > 0 {
        fallback_prefix.push(format!("-A{}", flags.after));
    }
    for (include, p) in &filters {
        fallback_prefix.push(format!(
            "--{}{p}",
            if *include { "include=" } else { "exclude=" }
        ));
    }
    for p in &exclude_dir {
        fallback_prefix.push(format!("--exclude-dir={p}"));
    }
    for p in &raw_patterns {
        fallback_prefix.push("-e".into());
        fallback_prefix.push(p.clone());
    }

    Ok(ParsedGrep {
        mode,
        flags,
        filters,
        exclude_dir,
        engine_patterns,
        operand_tokens,
        redirects,
        fallback_prefix,
    })
}

fn push_flag(argv: &mut Vec<String>, on: bool, flag: &str) {
    if on {
        argv.push(flag.to_string());
    }
}

// ── Pattern translation ───────────────────────────────────────────────────

/// Translate a BSD-dialect pattern to the engine dialect (Rust regex syntax).
/// Fail-closed: anything not provably equivalent falls back.
fn translate_pattern(pattern: &str, mode: MatchMode) -> Result<String, Fallback> {
    match mode {
        MatchMode::Fixed => Ok(pattern.to_string()),
        MatchMode::Extended => {
            check_ere_safe(pattern)?;
            Ok(pattern.to_string())
        }
        MatchMode::Basic => translate_bre(pattern),
    }
}

/// ERE passes through, but reject constructs whose engine semantics differ
/// from BSD (backrefs and unknown escapes are caught by compile validation).
fn check_ere_safe(pattern: &str) -> Result<(), Fallback> {
    let mut escaped = false;
    for c in pattern.chars() {
        if escaped {
            match c {
                '<' | '>' => return Err(Fallback::Pattern(r"\< \>".into())),
                's' | 'S' | 'w' | 'W' | 'd' | 'D' => {
                    return Err(Fallback::Pattern(r"\s\w\d".into()));
                }
                _ => {}
            }
            escaped = false;
            continue;
        }
        if c == '\\' {
            escaped = true;
        }
    }
    if pattern.contains("[[:<:]]") || pattern.contains("[[:>:]]") {
        return Err(Fallback::Pattern("word-boundary class".into()));
    }
    Ok(())
}

/// BRE → engine dialect. `\| \( \) \+ \? \{m,n\}` become metacharacters;
/// unescaped `| ( ) + ? { }` are literals (escaped); `\b` passes through;
/// backrefs, `\<`/`\>`, `\s`/`\w`/`\d` and unknown escapes fall back.
fn translate_bre(pattern: &str) -> Result<String, Fallback> {
    let mut out = String::with_capacity(pattern.len());
    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            let Some(esc) = chars.next() else {
                return Err(Fallback::Pattern("trailing backslash".into()));
            };
            match esc {
                '|' | '(' | ')' | '+' | '?' => out.push(esc),
                '{' => {
                    let mut body = String::new();
                    loop {
                        match chars.next() {
                            Some('\\') if chars.peek() == Some(&'}') => {
                                chars.next();
                                break;
                            }
                            Some(d @ '0'..='9') => body.push(d),
                            Some(',') => body.push(','),
                            Some(_) => {
                                return Err(Fallback::Pattern("bad interval".into()));
                            }
                            None => return Err(Fallback::Pattern("unclosed interval".into())),
                        }
                    }
                    out.push('{');
                    out.push_str(&body);
                    out.push('}');
                }
                '}' => return Err(Fallback::Pattern(r"lone \}".into())),
                'b' => out.push_str("\\b"),
                '1'..='9' => return Err(Fallback::Pattern("backreference".into())),
                '<' | '>' => return Err(Fallback::Pattern(r"\< \>".into())),
                's' | 'S' | 'w' | 'W' | 'd' | 'D' => {
                    return Err(Fallback::Pattern(r"\s\w\d".into()));
                }
                '.' | '*' | '\\' | '[' | ']' | '$' | '^' => {
                    out.push('\\');
                    out.push(esc);
                }
                other => return Err(Fallback::Pattern(format!(r"unknown escape \{other}"))),
            }
        } else {
            match c {
                '|' | '(' | ')' | '+' | '?' | '{' | '}' => {
                    // Unescaped → literal in BRE.
                    out.push('\\');
                    out.push(c);
                }
                _ => out.push(c),
            }
        }
    }
    Ok(out)
}

/// True when the pattern has unescaped alternation outside a character class.
/// BSD matching is POSIX leftmost-longest; the engine is leftmost-first, so
/// under `-o` any alternation can diverge on match length.
fn has_alternation(s: &str) -> bool {
    alternation_at(s, false)
}

fn has_top_level_alternation(s: &str) -> bool {
    alternation_at(s, true)
}

/// Scan for `|` outside classes, optionally only at paren depth zero.
fn alternation_at(s: &str, top_level_only: bool) -> bool {
    let mut depth = 0usize;
    let mut in_class = false;
    let mut escaped = false;
    for c in s.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' {
            escaped = true;
            continue;
        }
        if in_class {
            if c == ']' {
                in_class = false;
            }
            continue;
        }
        match c {
            '[' => in_class = true,
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            '|' if !top_level_only || depth == 0 => return true,
            _ => {}
        }
    }
    false
}

/// `-w` equivalence gate: `\<pat\>` ≡ `\b(?:pat)\b` only when every match
/// starts and ends with a word char. Conservative: the translated pattern
/// must start and end with a word char with no top-level alternation.
fn word_safe(pattern: &str) -> bool {
    let Some(first) = pattern.chars().next() else {
        return false;
    };
    let Some(last) = pattern.chars().last() else {
        return false;
    };
    if !is_word_char(first) || !is_word_char(last) {
        return false;
    }
    !has_top_level_alternation(pattern)
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

// ── Matcher construction (shared by parent validation and engine) ─────────

fn build_matcher(
    patterns: &[String],
    mode: MatchMode,
    flags: &GrepFlags,
) -> Result<grep_regex::RegexMatcher, grep_regex::Error> {
    let mut b = grep_regex::RegexMatcherBuilder::new();
    b.case_insensitive(flags.i)
        .word(flags.w)
        .whole_line(flags.x)
        .unicode(true);
    if mode == MatchMode::Fixed {
        b.fixed_strings(true);
    }
    b.build_many(patterns)
}

/// Matcher wrapper: BSD grep under C.UTF-8 treats any line containing invalid
/// UTF-8 as a silent non-match for every matcher mode (-F/-E/-i/-v alike) —
/// except in binary files, where a NUL in the first 32 KiB switches grep to
/// byte-oriented matching and invalid-UTF-8 match lines still count. The
/// wrapper rejects invalid haystacks when `validate_utf8` is set (text files)
/// and hides the line-terminator/fast-path hints, forcing the searcher's
/// line-by-line slow path (the only place per-line validation is sound).
#[derive(Clone)]
struct SearchMatcher {
    inner: grep_regex::RegexMatcher,
    validate_utf8: bool,
}

impl grep_matcher::Matcher for SearchMatcher {
    type Captures = <grep_regex::RegexMatcher as grep_matcher::Matcher>::Captures;
    type Error = <grep_regex::RegexMatcher as grep_matcher::Matcher>::Error;

    fn find_at(
        &self,
        haystack: &[u8],
        at: usize,
    ) -> Result<Option<grep_matcher::Match>, Self::Error> {
        if self.validate_utf8 && std::str::from_utf8(haystack).is_err() {
            return Ok(None);
        }
        self.inner.find_at(haystack, at)
    }

    fn new_captures(&self) -> Result<Self::Captures, Self::Error> {
        self.inner.new_captures()
    }

    fn line_terminator(&self) -> Option<grep_matcher::LineTerminator> {
        None // force the slow per-line path (see struct doc)
    }

    fn non_matching_bytes(&self) -> Option<&grep_matcher::ByteSet> {
        None // ditto
    }
}

// ── Operand resolution and the serve gate ─────────────────────────────────

/// Serve one grep segment: parse, translate, resolve operands, apply the
/// serve-worthiness gate, and build the spec + rewritten fragment.
fn serve_one_grep(
    segment: &str,
    verb: &str,
    cwd: &Path,
    home: &Path,
    allow_single: bool,
) -> Result<(EngineSpec, String), Fallback> {
    let mut words = grep_tokenize(segment)?;
    // The segment starts with the verb; the parser sees only its arguments.
    if words.first().is_some_and(|w| w.value == verb) {
        words.remove(0);
    } else {
        return Err(Fallback::NestedGrep);
    }
    let parsed = parse_grep_words(&words, verb)?;

    let mut operands: Vec<Operand> = Vec::new();
    if parsed.operand_tokens.is_empty() {
        if parsed.flags.r {
            // `grep -r pat` with no operands walks the cwd (BSD behavior).
            operands.push(Operand {
                display: ".".into(),
                resolved: cwd.to_string_lossy().into_owned(),
                trailing_slash: false,
            });
        } else {
            return Err(Fallback::StdinMode);
        }
    } else {
        for tok in &parsed.operand_tokens {
            if tok == "-" {
                return Err(Fallback::StdinMode);
            }
            let expanded = resolve_operand(tok, cwd, home)?;
            if expanded.is_empty() {
                return Err(Fallback::UnexpandableGlob);
            }
            operands.extend(expanded);
        }
    }

    // Serve gate: recursive walks (the expensive case), or multi-file
    // invocations. Single-file lookups stay on the real grep (faster).
    let mut dir_count = 0usize;
    for op in &operands {
        if fs::metadata(&op.resolved).is_ok_and(|m| m.is_dir()) {
            dir_count += 1;
        }
    }
    let serve = allow_single
        || (parsed.flags.r && (dir_count > 0 || operands.len() != 1))
        || operands.len() >= 2;
    if !serve {
        return Err(Fallback::SingleFile);
    }

    let mut fallback = parsed.fallback_prefix.clone();
    // A `-`-prefixed operand would be re-parsed as a flag by the exec'd grep;
    // `--` (which BSD grep honors after `-e` patterns) restores operand status.
    if operands.iter().any(|op| op.display.starts_with('-')) {
        fallback.push("--".into());
    }
    for op in &operands {
        fallback.push(op.display.clone());
    }

    let spec = EngineSpec {
        version: PROTOCOL_VERSION,
        verb: verb.to_string(),
        mode: parsed.mode,
        flags: parsed.flags,
        filters: parsed.filters,
        exclude_dir: parsed.exclude_dir,
        patterns: parsed.engine_patterns,
        operands,
        cwd: cwd.to_string_lossy().into_owned(),
        fallback,
    };

    let mut rewritten = build_rewritten(&spec);
    if !parsed.redirects.is_empty() {
        rewritten.push(' ');
        rewritten.push_str(&parsed.redirects.join(" "));
    }
    Ok((spec, rewritten))
}

/// Resolve one operand token into concrete operands (tilde, glob expansion,
/// relative-to-cwd). Unresolvable forms fall back.
fn resolve_operand(tok: &str, cwd: &Path, home: &Path) -> Result<Vec<Operand>, Fallback> {
    if tok == "~" {
        return Ok(vec![operand_from_path(
            &home.to_string_lossy(),
            home,
            false,
        )]);
    }
    if tok.starts_with('~') && !tok.starts_with("~/") {
        // `~user` home expansion is not statically resolvable — fail-closed.
        return Err(Fallback::UnresolvableOperand(tok.to_string()));
    }
    if has_unquoted_glob(tok) {
        // Includes `~/…` globs: expand_glob strips the `~/` itself.
        let pattern = unquote_word(tok)?;
        let matches = expand_glob(&pattern, cwd, home)?;
        if matches.is_empty() {
            return Err(Fallback::UnexpandableGlob);
        }
        return Ok(matches
            .iter()
            .map(|m| {
                let abs = if m.starts_with('/') {
                    PathBuf::from(m)
                } else {
                    cwd.join(m)
                };
                operand_from_path(m, &abs, false)
            })
            .collect());
    }
    if let Some(rest) = tok.strip_prefix("~/") {
        let expanded = home.join(rest);
        return Ok(vec![operand_from_path(
            &expanded.to_string_lossy(),
            &expanded,
            false,
        )]);
    }
    let value = unquote_word(tok)?;
    let trailing_slash = value.ends_with('/');
    let abs = if value.starts_with('/') {
        PathBuf::from(&value)
    } else {
        cwd.join(&value)
    };
    Ok(vec![operand_from_path(&value, &abs, trailing_slash)])
}

fn operand_from_path(display: &str, resolved: &Path, trailing_slash: bool) -> Operand {
    Operand {
        display: display.to_string(),
        resolved: resolved.to_string_lossy().into_owned(),
        trailing_slash,
    }
}

/// True when the raw token has glob metacharacters outside quotes/escapes.
fn has_unquoted_glob(tok: &str) -> bool {
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    for c in tok.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' {
            escaped = true;
            continue;
        }
        if c == '\'' && !in_double {
            in_single = !in_single;
            continue;
        }
        if c == '"' && !in_single {
            in_double = !in_double;
            continue;
        }
        if !in_single && matches!(c, '*' | '?' | '[') {
            return true;
        }
    }
    false
}

/// Bash-style glob expansion for operand tokens: dotfile exclusion, bracket
/// expressions, no-match → empty (the caller falls back). The returned
/// display strings match what the shell would pass to grep.
fn expand_glob(pattern: &str, cwd: &Path, home: &Path) -> Result<Vec<String>, Fallback> {
    let pattern = if let Some(rest) = pattern.strip_prefix("~/") {
        home.join(rest).to_string_lossy().into_owned()
    } else if pattern == "~" {
        home.to_string_lossy().into_owned()
    } else {
        pattern.to_string()
    };
    if pattern.ends_with('/') {
        return Err(Fallback::UnexpandableGlob);
    }
    let (base, display_prefix, comps) = if let Some(rest) = pattern.strip_prefix('/') {
        (PathBuf::from("/"), "/", rest.split('/').collect::<Vec<_>>())
    } else {
        (
            cwd.to_path_buf(),
            "",
            pattern.split('/').collect::<Vec<_>>(),
        )
    };
    let mut results = Vec::new();
    glob_walk(&base, display_prefix, &comps, &mut results);
    // The shell sorts glob expansions (LC_ALL=C.UTF-8 → byte order); grep
    // emits operands in command-line order, so operand order must match.
    results.sort();
    Ok(results)
}

fn glob_walk(dir: &Path, display: &str, comps: &[&str], results: &mut Vec<String>) {
    let Some((first, rest)) = comps.split_first() else {
        return;
    };
    if first.contains(['*', '?', '[']) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            // Bash glob semantics per component: fnmatch with FNM_PERIOD
            // (leading dot must match a literal dot).
            if !fnmatch_flags(first, &name, libc::FNM_PERIOD) {
                continue;
            }
            let child_display = join_display(display, &name);
            if rest.is_empty() {
                results.push(child_display);
            } else if entry.path().is_dir() {
                glob_walk(&entry.path(), &child_display, rest, results);
            }
        }
    } else {
        let next = dir.join(first);
        let child_display = join_display(display, first);
        if rest.is_empty() {
            if next.exists() {
                results.push(child_display);
            }
        } else if next.is_dir() {
            glob_walk(&next, &child_display, rest, results);
        }
    }
}

fn join_display(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else if prefix == "/" {
        format!("/{name}")
    } else {
        format!("{prefix}/{name}")
    }
}

// ── Engine side: hidden subcommand execution ──────────────────────────────

/// Entry point for the hidden `__grep-engine` subcommand (dispatched from
/// `main()` before instance-lock acquisition). `--probe` answers whether the
/// binary still carries the subcommand (self-update version skew). Any
/// runtime doubt `exec`s the real grep in place; when even that fails, the
/// sentinel exit code tells the parent to re-run the original command.
pub fn run_engine(args: &[String]) -> i32 {
    if args.first().map(String::as_str) == Some("--probe") {
        return 0;
    }
    // Rust ignores SIGPIPE by default; grep dies on a closed pipe (exit 141).
    // With the default disposition, `grep | head` would scan the whole input.
    // SAFETY: restoring the default SIGPIPE disposition is process-global and
    // idempotent; the engine process exists only to serve this one invocation.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    let Some(json) = args.first() else {
        eprintln!("grep: engine: missing spec");
        return ENGINE_FAILED_EXIT;
    };
    let spec: EngineSpec = match serde_json::from_str(json) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("grep: engine: bad spec: {e}");
            return ENGINE_FAILED_EXIT;
        }
    };
    if spec.version != PROTOCOL_VERSION {
        // Self-update swapped the binary mid-run: nothing is written yet, so
        // exec the real grep in place — its exit code cannot be masked by a
        // pipe member (unlike a sentinel exit).
        return exec_grep(&spec.fallback);
    }
    let fallback = spec.fallback.clone();
    // `serve` itself reports panics: nothing-written panics return Err (exec
    // grep in place); post-output panics return the sentinel. This outer
    // catch_unwind is the last-resort net for panics outside `serve`'s scope.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| serve(&spec)));
    match result {
        Ok(Ok(code)) => code,
        // Err only from pre-output checks (cwd mismatch, matcher build, or a
        // panic before anything was written) — exec'ing the real grep is clean.
        Ok(Err(())) => exec_grep(&fallback),
        // A panic escaped `serve` mid-search: partial output may have been
        // streamed; exec'ing grep would append to it (a false result). Exit
        // with the sentinel instead: the parent discards and re-runs.
        Err(_) => ENGINE_FAILED_EXIT,
    }
}

/// Replace the engine process with the real grep (PID/PGID preserved, so the
/// shell tool's process-group timeout kill still applies). Served invocations
/// never read stdin, so re-exec is always sound.
fn exec_grep(argv: &[String]) -> i32 {
    use std::os::unix::process::CommandExt;
    let Some(first) = argv.first() else {
        return ENGINE_FAILED_EXIT;
    };
    let err = std::process::Command::new(first).args(&argv[1..]).exec();
    eprintln!("grep: failed to exec {first}: {err}");
    ENGINE_FAILED_EXIT
}

/// Serve one spec: verify the cwd, then process each operand in order with
/// per-file grep semantics. Returns the aggregate exit code (0/1/2).
///
/// `Err(())` means nothing was written and the caller should exec the real
/// grep in place (cwd divergence, matcher build failure, or a panic before
/// any output). A panic after output started returns `Ok(ENGINE_FAILED_EXIT)`
/// — exec'ing grep then would append a false result, so the parent re-runs.
fn serve(spec: &EngineSpec) -> Result<i32, ()> {
    let actual_cwd = std::env::current_dir().map_err(|_| ())?;
    if actual_cwd != Path::new(&spec.cwd) {
        // The cd chain diverged at runtime (e.g. `;`-joined failing cd) —
        // the real grep in the actual cwd is the authentic result.
        return Err(());
    }
    let matcher = build_matcher(&spec.patterns, spec.mode, &spec.flags).map_err(|_| ())?;
    let mut out = Output::streaming();
    let code = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        serve_into(spec, &matcher, &mut out)
    }));
    match code {
        Ok(code) => {
            out.finish();
            Ok(code)
        }
        // A panic before anything was written: exec'ing the real grep is
        // clean, and its exit code cannot be masked by a pipe member.
        Err(_) if out.written == 0 => Err(()),
        // A panic mid-search may have streamed partial output; exec'ing grep
        // would append to it. Flush and return the sentinel: the parent
        // discards this run and re-runs the original command.
        Err(_) => {
            out.finish();
            Ok(ENGINE_FAILED_EXIT)
        }
    }
}

/// Serve a spec writing into caller-owned buffers (testable in-process).
fn serve_into(spec: &EngineSpec, matcher: &grep_regex::RegexMatcher, out: &mut Output) -> i32 {
    let show_prefix = !spec.flags.h && (spec.flags.H || spec.flags.r || spec.operands.len() > 1);
    // `max` keeps Error over Match/NoMatch — any error yields exit 2 even when
    // matches were printed (BSD), mirroring walk_dir's aggregation.
    let mut result = OperandResult::NoMatch;
    for op in &spec.operands {
        result = result.max(process_operand(op, spec, matcher, out, show_prefix));
    }
    match result {
        OperandResult::NoMatch => 1,
        OperandResult::Match => 0,
        OperandResult::Error => 2,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum OperandResult {
    NoMatch,
    Match,
    Error,
}

/// Process one operand (explicit file/dir or walk root) with BSD symlink and
/// error semantics, emitting output through `out`.
fn process_operand(
    op: &Operand,
    spec: &EngineSpec,
    matcher: &grep_regex::RegexMatcher,
    out: &mut Output,
    show_prefix: bool,
) -> OperandResult {
    let path = Path::new(&op.resolved);
    let display = &op.display;

    let lmeta = match fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) => {
            emit_error(spec, out, display, &e.to_string());
            return OperandResult::Error;
        }
    };

    if lmeta.file_type().is_symlink() {
        if spec.flags.r && !op.trailing_slash {
            return OperandResult::NoMatch; // silent skip
        }
        match fs::metadata(path) {
            Err(e) => {
                emit_error(spec, out, display, &e.to_string());
                return OperandResult::Error;
            }
            Ok(meta) => {
                if meta.is_dir() {
                    if !spec.flags.r {
                        emit_error(spec, out, display, "Is a directory");
                        return OperandResult::Error;
                    }
                    if dir_excluded(display, &spec.exclude_dir) {
                        return OperandResult::NoMatch;
                    }
                    return walk_dir(op, spec, matcher, out, show_prefix);
                }
                if op.trailing_slash {
                    emit_error(spec, out, display, "Not a directory");
                    return OperandResult::Error;
                }
                if !file_allowed_by_filters(display, spec) {
                    return OperandResult::NoMatch;
                }
                return search_file(path, display, spec, matcher, out, show_prefix);
            }
        }
    }

    if lmeta.is_dir() {
        if dir_excluded(display, &spec.exclude_dir) {
            return OperandResult::NoMatch;
        }
        if !spec.flags.r {
            emit_error(spec, out, display, "Is a directory");
            return OperandResult::Error;
        }
        return walk_dir(op, spec, matcher, out, show_prefix);
    }

    if op.trailing_slash {
        emit_error(spec, out, display, "Not a directory");
        return OperandResult::Error;
    }
    // Explicit file operands are always searched, subject to the BSD
    // --include/--exclude filters (which apply to explicit operands too).
    if !file_allowed_by_filters(display, spec) {
        return OperandResult::NoMatch;
    }
    search_file(path, display, spec, matcher, out, show_prefix)
}

/// BSD fnmatch semantics for --include/--exclude: patterns match the basename
/// OR the full traversal path, `*` crosses `/`, anchored full-string, and the
/// LAST matching pattern in command-line order decides (include vs exclude).
fn file_allowed_by_filters(display: &str, spec: &EngineSpec) -> bool {
    // One ordered list of include+exclude patterns; last match wins. Each
    // pattern matches the basename or the full traversal path.
    let base = display.rsplit('/').next().unwrap_or(display);
    let mut last: Option<bool> = None;
    for (include, pat) in &spec.filters {
        if fnmatch(pat, display) || fnmatch(pat, base) {
            last = Some(*include);
        }
    }
    match last {
        Some(include) => include,
        // No pattern matched: with any --include present the file is skipped.
        None => !spec.filters.iter().any(|(include, _)| *include),
    }
}

/// `--exclude-dir` filter (basename or full traversal path).
fn dir_excluded(display: &str, exclude_dir: &[String]) -> bool {
    let base = display.rsplit('/').next().unwrap_or(display);
    exclude_dir
        .iter()
        .any(|pat| fnmatch(pat, base) || fnmatch(pat, display))
}

/// Host fnmatch; `flags` mirror bash globbing (FNM_PERIOD excludes dotfiles)
/// or BSD grep's --include/--exclude calls (0, no pathname semantics).
fn fnmatch_flags(pattern: &str, s: &str, flags: i32) -> bool {
    let Ok(p) = std::ffi::CString::new(pattern) else {
        return false;
    };
    let Ok(n) = std::ffi::CString::new(s) else {
        return false;
    };
    // SAFETY: both CStrings are NUL-free and outlive the call.
    unsafe { libc::fnmatch(p.as_ptr(), n.as_ptr(), flags) == 0 }
}

/// fnmatch without FNM_PATHNAME — exactly what BSD grep calls:
/// `*`/`?` cross `/`, `[...]` classes (incl. POSIX names), anchored full-string.
fn fnmatch(pattern: &str, s: &str) -> bool {
    fnmatch_flags(pattern, s, 0)
}

/// Recursive walk with rg-default exclusions (hidden, gitignore, .ignore, git
/// exclude, global gitignore) in walkdir order, plus BSD include/exclude
/// filters and symlink skipping. Explicit operands never pass through here.
fn walk_dir(
    op: &Operand,
    spec: &EngineSpec,
    matcher: &grep_regex::RegexMatcher,
    out: &mut Output,
    show_prefix: bool,
) -> OperandResult {
    let root_abs = PathBuf::from(&op.resolved);
    let root_display = op.display.clone();
    let exclude_dir = spec.exclude_dir.clone();
    let root_for_filter = root_abs.clone();
    let mut result = OperandResult::NoMatch;

    let mut builder = ignore::WalkBuilder::new(&root_abs);
    builder.filter_entry(move |entry| {
        if entry.file_type().is_some_and(|t| t.is_dir()) && entry.depth() > 0 {
            let display = traversal_display(&root_display, &root_for_filter, entry.path());
            !dir_excluded(&display, &exclude_dir)
        } else {
            true
        }
    });
    for entry in builder.build() {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                // Unreadable dir/file during traversal → grep-style error.
                let (path, message) = walk_error_info(&e);
                let display = path.as_deref().map_or_else(
                    || op.display.clone(),
                    |p| traversal_display(&op.display, &root_abs, p),
                );
                emit_error(spec, out, &display, &message);
                result = OperandResult::Error;
                continue;
            }
        };
        let ft = entry.file_type();
        if ft.is_some_and(|t| t.is_symlink() || t.is_dir()) {
            continue; // grep -r skips symlinks; dirs are traversed by the walk
        }
        let display = traversal_display(&op.display, &root_abs, entry.path());
        if !file_allowed_by_filters(&display, spec) {
            continue;
        }
        // Any error yields exit 2 even when matches were printed (BSD);
        // `max` keeps Error over Match/NoMatch.
        let r = search_file(entry.path(), &display, spec, matcher, out, show_prefix);
        result = result.max(r);
    }
    result
}

/// Display path for a walked entry: the operand spelling + relative suffix.
fn traversal_display(root_display: &str, root_abs: &Path, entry: &Path) -> String {
    let rel = entry.strip_prefix(root_abs).unwrap_or(entry);
    let rel = rel.to_string_lossy();
    if rel.is_empty() {
        root_display.to_string()
    } else {
        format!("{}/{}", root_display.trim_end_matches('/'), rel)
    }
}

/// Extract (path, normalized message) from an ignore walk error.
fn walk_error_info(e: &ignore::Error) -> (Option<PathBuf>, String) {
    if let ignore::Error::WithPath { path, err } = e {
        (Some(path.clone()), normalize_io_message(err))
    } else {
        (None, normalize_io_message(e))
    }
}

/// Strip the ` (os error N)` suffix from io error messages so the phrasing
/// matches BSD grep's.
fn strip_os_error(msg: &str) -> &str {
    match msg.split_once(" (os error ") {
        Some((head, tail)) if tail.ends_with(')') => head,
        _ => msg,
    }
}

fn normalize_io_message(e: &ignore::Error) -> String {
    let msg = e
        .io_error()
        .map_or_else(|| e.to_string(), ToString::to_string);
    strip_os_error(&msg).to_string()
}

/// Search one file with BSD binary semantics (32 KiB NUL detection; binary
/// files print the `Binary file … matches` message instead of lines unless
/// `-a`; matches anywhere in the file count for the exit code, and matching is
/// byte-oriented — invalid-UTF-8 match lines are not poisoned).
fn search_file(
    path: &Path,
    display: &str,
    spec: &EngineSpec,
    matcher: &grep_regex::RegexMatcher,
    out: &mut Output,
    show_prefix: bool,
) -> OperandResult {
    let mut file = match fs::File::open(path) {
        Ok(f) => f,
        Err(e) => {
            emit_error(spec, out, display, &e.to_string());
            return OperandResult::Error;
        }
    };
    let mut head = vec![0u8; BINARY_WINDOW];
    let n = match file.read(&mut head) {
        Ok(n) => n,
        Err(e) => {
            emit_error(spec, out, display, &e.to_string());
            return OperandResult::Error;
        }
    };
    let binary = head[..n].contains(&0);
    // Non-seekable operands (FIFOs, /dev/*) cannot be re-read; stream the
    // buffered head first, then the remainder — BSD grep likewise never seeks
    // (it reads the window and continues from the stream).
    let mut input = io::Cursor::new(&head[..n]).chain(file);

    let search_matcher = SearchMatcher {
        inner: matcher.clone(),
        validate_utf8: !binary,
    };
    let mut sink = GrepSink {
        spec,
        display,
        show_prefix,
        binary: binary && !spec.flags.a,
        message_emitted: false,
        selected_any: false,
        count: 0,
        matcher: &search_matcher,
        out,
    };
    let matcher_for_search = search_matcher.clone();
    let mut sb = grep_searcher::SearcherBuilder::new();
    sb.line_number(spec.flags.n)
        .invert_match(spec.flags.v)
        .binary_detection(grep_searcher::BinaryDetection::none())
        .heap_limit(Some(HEAP_LIMIT));
    let count_mode = spec.flags.count_mode();
    if !count_mode && spec.flags.before > 0 {
        sb.before_context(spec.flags.before);
    }
    if !count_mode && spec.flags.after > 0 {
        sb.after_context(spec.flags.after);
    }
    if let Some(m) = spec.flags.m {
        sb.max_matches(Some(m));
    }
    let mut searcher = sb.build();
    if let Err(e) = searcher.search_reader(matcher_for_search, &mut input, &mut sink) {
        emit_error(spec, out, display, &e.to_string());
        return OperandResult::Error;
    }
    if count_mode {
        sink.finish_count();
    }
    if sink.selected_any {
        OperandResult::Match
    } else {
        OperandResult::NoMatch
    }
}

fn emit_error(spec: &EngineSpec, out: &mut Output, display: &str, message: &str) {
    if !spec.flags.s {
        // io errors carry a " (os error N)" suffix; BSD grep prints the bare
        // message ("No such file or directory", "Permission denied", …).
        out.write_err(&format!(
            "{}: {display}: {}\n",
            spec.verb,
            strip_os_error(message)
        ));
    }
}

// ── Sink: BSD-grep-compatible output formatting ───────────────────────────

#[allow(clippy::struct_excessive_bools)] // grep CLI flag surface
struct GrepSink<'a> {
    spec: &'a EngineSpec,
    display: &'a str,
    show_prefix: bool,
    binary: bool,
    message_emitted: bool,
    selected_any: bool,
    /// Selected-line count for -c/-l (capped by -m, or at 1 by the -l stop).
    count: u64,
    matcher: &'a SearchMatcher,
    out: &'a mut Output,
}

impl grep_searcher::Sink for GrepSink<'_> {
    type Error = io::Error;

    fn matched(
        &mut self,
        _searcher: &grep_searcher::Searcher,
        mat: &grep_searcher::SinkMatch<'_>,
    ) -> Result<bool, io::Error> {
        if self.spec.flags.count_mode() {
            // Count selected lines; -l stops at the first one. Binary files
            // get the plain count/name (no "Binary file" message), and
            // byte-oriented matching still counts invalid-UTF-8 lines.
            self.selected_any = true;
            self.count += 1;
            return Ok(!self.spec.flags.l);
        }
        if self.binary {
            if !self.message_emitted {
                self.message_emitted = true;
                self.out
                    .write_bytes(format!("Binary file {} matches\n", self.display).as_bytes());
            }
            self.selected_any = true;
            return Ok(false); // existence is enough
        }
        self.selected_any = true;
        let content = trim_line_terminator(mat.bytes());
        if self.spec.flags.o {
            // BSD -o never prints zero-length matches.
            let mut matches = Vec::new();
            let _ = self.matcher.find_iter(content, |m| {
                if !m.is_empty() {
                    matches.push((m.start(), m.end()));
                }
                true
            });
            for (start, end) in matches {
                self.write_prefix(mat.line_number(), false);
                self.out.write_bytes(&content[start..end]);
                self.out.write_byte(b'\n');
            }
        } else {
            self.write_prefix(mat.line_number(), false);
            self.out.write_bytes(content);
            self.out.write_byte(b'\n');
        }
        self.out.flush();
        Ok(true)
    }

    fn context(
        &mut self,
        _searcher: &grep_searcher::Searcher,
        ctx: &grep_searcher::SinkContext<'_>,
    ) -> Result<bool, io::Error> {
        // Count mode never configures context (search_file), so no events here.
        if self.binary {
            return Ok(true); // binary files only emit the message
        }
        let content = trim_line_terminator(ctx.bytes());
        self.write_prefix(ctx.line_number(), true);
        self.out.write_bytes(content);
        self.out.write_byte(b'\n');
        self.out.flush();
        Ok(true)
    }

    fn context_break(&mut self, _searcher: &grep_searcher::Searcher) -> Result<bool, io::Error> {
        self.out.write_bytes(b"--\n");
        Ok(true)
    }
}

impl GrepSink<'_> {
    /// Write the line prefix: `path` + separator + `lineno` + separator.
    /// Match lines use `:` (path separator replaced by NUL with --null);
    /// context lines use `-` (the path separator is still NUL with --null).
    fn write_prefix(&mut self, lineno: Option<u64>, is_context: bool) {
        if self.show_prefix {
            self.out.write_bytes(self.display.as_bytes());
            let sep = if self.spec.flags.null {
                b'\0'
            } else if is_context {
                b'-'
            } else {
                b':'
            };
            self.out.write_byte(sep);
        }
        if let Some(n) = lineno {
            self.out.write_bytes(n.to_string().as_bytes());
            self.out.write_byte(if is_context { b'-' } else { b':' });
        }
    }

    /// Emit the -c count line and/or the -l name line after the scan. BSD
    /// order: count first, then the name (combined -c -l). Zero-count files
    /// always print their count; -l prints the name only when at least one
    /// line was selected. --null NUL-terminates the name only; the count line
    /// keeps the `path:count\n` shape.
    fn finish_count(&mut self) {
        if self.spec.flags.c {
            if self.show_prefix {
                self.out.write_bytes(self.display.as_bytes());
                self.out.write_byte(b':');
            }
            self.out.write_bytes(self.count.to_string().as_bytes());
            self.out.write_byte(b'\n');
        }
        if self.spec.flags.l && self.selected_any {
            self.out.write_bytes(self.display.as_bytes());
            self.out
                .write_byte(if self.spec.flags.null { 0 } else { b'\n' });
        }
        self.out.flush();
    }
}

fn trim_line_terminator(b: &[u8]) -> &[u8] {
    b.strip_suffix(b"\n").unwrap_or(b)
}

// ── Output: bounded, self-capping writer ──────────────────────────────────

/// Streams matched lines to stdout as they are produced (so `head` early-exit
/// propagates via SIGPIPE like the real grep), self-capping at [`OUTPUT_CAP`]
/// bytes after which writing stops but the search continues (exit codes stay
/// correct for huge outputs). With SIGPIPE at its default disposition a write
/// to a closed pipe kills the process — exactly like grep. Stderr is buffered
/// (small, rare) and flushed on finish.
struct Output {
    sink: OutputSink,
    err: Vec<u8>,
    written: usize,
    capped: bool,
}

enum OutputSink {
    Stdout(io::BufWriter<io::Stdout>),
    // In-memory sink used only by the macOS-gated parity tests.
    #[cfg_attr(not(all(test, target_os = "macos")), allow(dead_code))]
    Buffer(Vec<u8>),
}

impl Output {
    /// Production sink: streams to the process stdout.
    fn streaming() -> Self {
        Output {
            sink: OutputSink::Stdout(io::BufWriter::with_capacity(16 * 1024, io::stdout())),
            err: Vec::new(),
            written: 0,
            capped: false,
        }
    }

    /// In-memory sink for the differential parity tests.
    #[cfg_attr(not(all(test, target_os = "macos")), allow(dead_code))]
    fn buffered() -> Self {
        Output {
            sink: OutputSink::Buffer(Vec::new()),
            err: Vec::new(),
            written: 0,
            capped: false,
        }
    }

    fn write_bytes(&mut self, b: &[u8]) {
        if self.capped {
            return;
        }
        let take = (OUTPUT_CAP - self.written).min(b.len());
        match &mut self.sink {
            OutputSink::Stdout(w) => {
                let _ = w.write_all(&b[..take]);
            }
            OutputSink::Buffer(v) => v.extend_from_slice(&b[..take]),
        }
        self.written += take;
        if self.written >= OUTPUT_CAP {
            self.capped = true;
        }
    }

    fn write_byte(&mut self, b: u8) {
        self.write_bytes(&[b]);
    }

    /// Error message (stderr) — bounded.
    fn write_err(&mut self, s: &str) {
        if self.err.len() < 64 * 1024 {
            self.err.extend_from_slice(s.as_bytes());
        }
    }

    /// Push buffered stdout to the pipe (per line) so `head` sees data and
    /// its early exit reaches us via SIGPIPE.
    fn flush(&mut self) {
        if let OutputSink::Stdout(w) = &mut self.sink {
            let _ = w.flush();
        }
    }

    /// Flush stdout and emit the buffered stderr.
    fn finish(self) {
        if let OutputSink::Stdout(mut w) = self.sink {
            let _ = w.flush();
        }
        let stderr = io::stderr();
        let mut lock = stderr.lock();
        let _ = lock.write_all(&self.err);
        let _ = lock.flush();
    }
}

// ── Differential parity matrix (macOS-gated: the parity target is the host
//    BSD grep; on other Unix hosts the system grep differs) ────────────────

#[cfg(all(test, target_os = "macos"))]
mod parity_tests {
    use super::*;
    use std::process::Command;

    /// Build the fixture tree used by every matrix row.
    fn build_fixture(ws: &Path, home: &Path) {
        // Text files.
        fs::write(ws.join("a.txt"), "foo\nbar\nfoo\nbaz\n").unwrap();
        fs::write(ws.join("b.txt"), "qux\nfoo\n").unwrap();
        fs::write(ws.join("c.txt"), "apple\n").unwrap();
        fs::write(ws.join("m.txt"), "a\nb\na\nb\na\n").unwrap();
        fs::write(ws.join("ctx.txt"), "a\nb\nc\nd\ne\nf\ng\nh\ni\n").unwrap();
        fs::write(ws.join("w1.txt"), "foo.bar\nfoo x\n").unwrap();
        fs::write(ws.join("x1.txt"), "foo\n^foo\n").unwrap();
        fs::write(ws.join("z2.txt"), "ab\n\ncd\n").unwrap();
        fs::write(ws.join("uni.txt"), "café\nCAFÉ\n").unwrap();
        fs::write(ws.join("uni2.txt"), "straße\nstrasse\n").unwrap();
        fs::write(ws.join("inv.txt"), b"ok\nbad\xffline\nok2\n").unwrap();
        fs::write(ws.join("br.txt"), "foo\nbar\nfoobar\nfooXbar\n").unwrap();
        fs::write(ws.join("br2.txt"), "a.b\naab\n").unwrap();
        fs::write(ws.join("dash.txt"), "-x1\n").unwrap();
        fs::write(ws.join("-x1"), "x\n").unwrap();
        fs::write(ws.join("pipe.txt"), "a|b\nx\na|b\n").unwrap();
        fs::write(ws.join("xx.txt"), "x1\nx2\n").unwrap();
        // Plain walk tree (no gitignore, no dotfiles → full parity).
        fs::create_dir_all(ws.join("plain/d1")).unwrap();
        fs::write(ws.join("plain/a.txt"), "x1\n").unwrap();
        fs::write(ws.join("plain/d1/b.txt"), "x2\n").unwrap();
        fs::write(ws.join("plain/d1/c.txt"), "x3\n").unwrap();
        fs::write(ws.join("plain/e.txt"), "x5\n").unwrap();
        fs::create_dir_all(ws.join("plain/d2")).unwrap();
        fs::write(ws.join("plain/d2/d.txt"), "x4\n").unwrap();
        // Exclusion-delta tree: hidden + gitignored content (needs a .git dir
        // for gitignore rules to apply).
        fs::create_dir_all(ws.join("ign/.git")).unwrap();
        fs::write(ws.join("ign/visible.txt"), "x1\n").unwrap();
        fs::write(ws.join("ign/.hidden.txt"), "x2\n").unwrap();
        fs::write(ws.join("ign/.gitignore"), "*.log\n").unwrap();
        fs::write(ws.join("ign/skip.log"), "x3\n").unwrap();
        // Binary files.
        fs::create_dir_all(ws.join("bindir")).unwrap();
        fs::write(ws.join("bindir/bin1.dat"), b"hello\x00world\nneedle\n").unwrap();
        fs::write(ws.join("bindir/bin2.dat"), b"needle\x00world\n").unwrap();
        fs::write(ws.join("bindir/bin3.dat"), b"hello\x00world\n").unwrap();
        fs::write(ws.join("bindir/bin4.dat"), b"needle\nhello\x00world\n").unwrap();
        // NUL in the window + an invalid-UTF-8 match line: BSD switches to
        // byte-oriented matching, so the match still counts (message/exit 0,
        // raw line under -a) instead of being poisoned by the line validation.
        fs::write(ws.join("bindir/bin5.dat"), b"\xff\x00needle\n").unwrap();
        // Invalid-UTF-8 line, no match: silent exit 1.
        fs::write(ws.join("bindir/bin6.dat"), b"\xff\x00other\n").unwrap();
        // Symlinks.
        std::os::unix::fs::symlink("a.txt", ws.join("filelink")).unwrap();
        std::os::unix::fs::symlink("plain", ws.join("dirlink")).unwrap();
        std::os::unix::fs::symlink("plain", ws.join("plain/dirlink2")).unwrap();
        std::os::unix::fs::symlink("missing", ws.join("broken")).unwrap();
        // Home-dir tree for ~ operands.
        fs::create_dir_all(home.join("htree/sub")).unwrap();
        fs::write(home.join("htree/f1.txt"), "needle\n").unwrap();
        fs::write(home.join("htree/sub/f2.txt"), "needle\n").unwrap();
        // Subdirectory for cd chains.
        fs::create_dir_all(ws.join("sub")).unwrap();
        fs::write(ws.join("sub/s.txt"), "needle\n").unwrap();
    }

    /// Run the engine in-process on one spec; returns (stdout, stderr, code).
    fn engine_run(spec: &EngineSpec) -> (Vec<u8>, Vec<u8>, i32) {
        let matcher =
            build_matcher(&spec.patterns, spec.mode, &spec.flags).expect("matcher builds");
        let mut out = Output::buffered();
        let code = serve_into(spec, &matcher, &mut out);
        let Output {
            sink: OutputSink::Buffer(buf),
            err,
            ..
        } = out
        else {
            unreachable!("buffered sink")
        };
        (buf, err, code)
    }

    /// Run the ORIGINAL command through a real shell (the authentic reference —
    /// shell glob expansion, `cd` chains, pipes and BSD grep all apply).
    fn real_run_shell(command: &str, cwd: &Path, home: &Path) -> (Vec<u8>, Vec<u8>, i32) {
        let out = Command::new("/bin/sh")
            .arg("-c")
            .arg(command)
            .current_dir(cwd)
            .env("LC_ALL", "C.UTF-8")
            .env("HOME", home)
            .output()
            .expect("system grep runs");
        (
            out.stdout,
            out.stderr,
            out.status.code().expect("grep exits normally"),
        )
    }

    /// Run the real grep with the spec's fallback argv (the grep member only —
    /// used for pipeline rows, whose head member the e2e bench covers end-to-end).
    fn real_run_fallback(spec: &EngineSpec) -> (Vec<u8>, Vec<u8>, i32) {
        let bin = match spec.verb.as_str() {
            "egrep" => "/usr/bin/egrep",
            "fgrep" => "/usr/bin/fgrep",
            _ => "/usr/bin/grep",
        };
        let out = Command::new(bin)
            .args(&spec.fallback[1..])
            .current_dir(&spec.cwd)
            .env("LC_ALL", "C.UTF-8")
            .output()
            .expect("system grep runs");
        (
            out.stdout,
            out.stderr,
            out.status.code().expect("grep exits normally"),
        )
    }

    /// Assert the engine's output/exit are byte-identical to the real grep.
    fn assert_parity(command: &str, ws: &Path, home: &Path) {
        let (specs, rewritten) = analyze_command(command, ws, home, true)
            .unwrap_or_else(|e| panic!("{command}: expected servable, got {e}"));
        assert_eq!(specs.len(), 1, "{command}: expected one grep member");
        let (eout, eerr, ecode) = engine_run(&specs[0]);
        let piped = command.split_whitespace().any(|w| w == "|");
        let (rout, rerr, rcode) = if piped {
            assert!(
                rewritten.contains(" | "),
                "{command}: head member not preserved"
            );
            real_run_fallback(&specs[0])
        } else {
            real_run_shell(command, ws, home)
        };
        assert_eq!(eout, rout, "stdout mismatch for {command}");
        assert_eq!(eerr, rerr, "stderr mismatch for {command}");
        assert_eq!(ecode, rcode, "exit mismatch for {command}");
    }

    /// Assert the command falls back (original command untouched).
    fn assert_falls_back(command: &str, ws: &Path, home: &Path) {
        assert!(
            analyze_command(command, ws, home, false).is_err(),
            "{command}: expected fallback, got served"
        );
    }

    #[test]
    fn differential_parity_matrix() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let ws = tmp.path().join("ws");
        let home = tmp.path().join("home");
        fs::create_dir_all(&ws).expect("ws");
        fs::create_dir_all(&home).expect("home");
        build_fixture(&ws, &home);

        let rows: &[&str] = &[
            // ── BRE translation ──
            "grep -n 'foo\\|bar' a.txt",
            "grep -n '\\(foo\\)\\?bar' br.txt",
            "grep -n 'fo\\+o' br.txt",
            "grep -n 'fo\\?o' br.txt",
            "grep -n 'fo\\{2,3\\}' br.txt",
            "grep -n 'foo|bar' br.txt",
            "grep -n 'a(b' br2.txt",
            "grep -n 'a+b' br2.txt",
            "grep -n 'a.b' br2.txt",
            "grep -n 'a\\.b' br2.txt",
            "grep -n 'x1$' plain/a.txt",
            "grep -n '^x' plain/a.txt",
            "grep -n 'cat\\b' br.txt",
            "grep -n '[[:digit:]]' br.txt",
            "grep -n 'fo\\{2,\\}' br.txt",
            // ── ERE / fixed ──
            "egrep -n 'foo|bar' br.txt",
            "egrep -n '(foo)+' br.txt",
            "egrep -n 'a{2}' br2.txt",
            "fgrep -n 'a.b' br2.txt",
            "fgrep -n 'foo' a.txt",
            // ── Output shapes ──
            "grep -n x a.txt b.txt",
            "grep -H x a.txt",
            "grep -h x a.txt b.txt",
            "grep -r x plain",
            "grep -rn x plain",
            "grep -rn x plain/e.txt plain/d1",
            "grep -rn --null x plain",
            "grep -Hn --null x a.txt b.txt",
            "grep -o x a.txt b.txt",
            "grep -on o a.txt",
            "grep -oH x a.txt",
            "grep -o '' z2.txt a.txt",
            "fgrep -o 'a|b' pipe.txt a.txt", // fixed mode: literal `|` stays served
            "grep -n -A1 -B1 d ctx.txt",
            "grep -A1 -B1 'd\\|h' ctx.txt",
            "grep -A1 -B1 d a.txt b.txt",
            "grep -n -A1 b a.txt",
            // ── -v / -i / unicode / invalid UTF-8 ──
            "grep -v foo a.txt",
            "grep -v -o foo a.txt",
            "grep -vn foo a.txt b.txt",
            "grep -v '' z2.txt a.txt", // empty pattern matches all → inverted: nothing
            "grep -i 'café' uni.txt",
            "grep -i 'straße' uni2.txt",
            "grep -i 'STRASSE' uni2.txt",
            "grep 'ok' inv.txt",
            "grep -v 'ok' inv.txt",
            "grep -E 'b.d' inv.txt",
            "grep -i 'BAD' inv.txt",
            "grep -F 'bad' inv.txt",
            // ── -w / -x ──
            "grep -w foo a.txt",
            "grep -w café uni.txt",
            "grep -x foo x1.txt",
            "grep -x '^foo' x1.txt",
            "grep -x '' z2.txt",
            "grep -x -e foo -e bar a.txt",
            "grep -iw Foo a.txt",
            // ── -m ──
            "grep -m2 a m.txt",
            "grep -m2 -v a m.txt",
            "grep -m2 a m.txt c.txt",
            "grep -m1 -A1 a m.txt",
            "grep -m1 -B1 b m.txt",
            // ── -c (count) ──
            "grep -c foo a.txt",
            "grep -c foo a.txt b.txt",
            "grep -c zzz a.txt b.txt",  // zero counts still print, exit 1
            "grep -ch foo a.txt b.txt", // -h suppresses -c prefixes
            "grep -cH foo a.txt",       // -H forces the prefix
            "grep -cn foo a.txt b.txt", // -n ignored under -c
            "grep -c -A1 foo a.txt b.txt", // context accepted but ignored
            "grep -cv -A1 foo a.txt b.txt", // -v+context gate relaxed under -c
            "grep -co foo a.txt b.txt", // -o inert under -c
            "grep -co -m2 foo a.txt b.txt", // -m+-o gate relaxed under -c
            "grep -co 'foo\\|bar' a.txt b.txt", // -o+alternation gate relaxed under -c
            "grep -ci FOO a.txt b.txt",
            "grep -cw foo a.txt b.txt",
            "grep -cx foo x1.txt a.txt",
            "grep -c -e foo -e bar a.txt b.txt",
            "egrep -c 'foo|bar' a.txt b.txt",
            "fgrep -c 'a.b' br2.txt a.txt",
            "grep -c -m2 a m.txt c.txt",    // -m caps the count
            "grep -c -m10 foo a.txt b.txt", // -m above the match count: full count
            "grep -c -m2 -v a m.txt",
            "grep -cv foo a.txt b.txt",
            "grep -cvo foo a.txt b.txt", // -v -o normalizes to plain -v under -c
            "grep -c '' z2.txt a.txt",   // empty pattern matches all
            "grep -c foo inv.txt",       // invalid-UTF-8 line silently non-matching
            "grep -c bad inv.txt",
            "grep -cv foo inv.txt",
            "grep -cr x plain",
            "grep -chr x plain", // -h on a recursive -c
            "grep -crx x1 plain/a.txt plain/e.txt",
            "grep -cr x plain missing.txt", // counts still print, exit 2
            // ── -l (files-with-matches) ──
            "grep -l foo a.txt",
            "grep -l foo a.txt b.txt c.txt",
            "grep -l zzz a.txt b.txt",  // no names, exit 1
            "grep -lh foo a.txt b.txt", // -h has no effect on -l names
            "grep -lH foo a.txt",
            "grep -ln foo a.txt b.txt",
            "grep -l -B1 foo a.txt b.txt", // context accepted but ignored
            "grep -lw foo a.txt b.txt",
            "grep -lv a m.txt b.txt",
            "grep -lx foo x1.txt a.txt",
            "grep -l -e foo -e bar a.txt b.txt",
            "grep -lr x plain",
            // ── combined -c -l (BSD: count capped at 1 + name) ──
            "grep -cl foo a.txt",
            "grep -cl foo a.txt b.txt c.txt",
            "grep -cl zzz a.txt b.txt", // zero-count files print only the count
            "grep -clH foo a.txt",
            "grep -clv a m.txt b.txt",
            "grep -cl -m5 a m.txt", // -l caps the combined count at 1
            "grep -cla needle bindir/bin1.dat bindir/bin2.dat bindir/bin3.dat",
            "grep -cl -e foo -e bar a.txt b.txt",
            "grep -clr x plain",
            "grep -clh --null foo a.txt b.txt",
            // ── --null: NUL-terminates -l names only ──
            "grep -l --null foo a.txt b.txt",
            "grep -c --null foo a.txt b.txt",
            "grep -cl --null foo a.txt b.txt",
            "grep -lr --null x plain",
            "grep -cr --null x plain",
            "grep -clr --null x plain",
            // ── Errors and exit codes ──
            "grep x missing.txt a.txt",
            "grep -s x missing.txt a.txt",
            "grep x plain a.txt",
            "grep x xx.txt/ a.txt",
            "grep -rn x plain missing.txt",
            // ── Binary files ──
            "grep -rn needle bindir",
            "grep -r needle bindir",
            "grep -rn -a needle bindir/bin1.dat bindir/bin4.dat",
            "grep -rn -v needle bindir",
            "grep -rn -o needle bindir",
            // -c/-l on binary files: plain count/name, no "Binary file" message
            "grep -c needle bindir/bin1.dat bindir/bin2.dat bindir/bin3.dat",
            "grep -l needle bindir/bin1.dat bindir/bin2.dat bindir/bin3.dat",
            "grep -cl needle bindir/bin1.dat bindir/bin2.dat bindir/bin3.dat",
            "grep -cv needle bindir/bin1.dat bindir/bin2.dat bindir/bin3.dat",
            "grep -clv --null needle bindir/bin1.dat bindir/bin2.dat bindir/bin3.dat",
            "grep -ca needle bindir/bin1.dat bindir/bin2.dat bindir/bin3.dat",
            "grep -cr needle bindir",
            "grep -lr needle bindir",
            "grep -clr needle bindir",
            "grep -cs x missing.txt a.txt", // -s suppresses stderr, counts print, exit 2
            // Binary + invalid-UTF-8 match line: byte-oriented detection must
            // count the match (Binary message / exit 0), and -a must print the
            // raw line — the per-line UTF-8 gate must not poison it.
            "grep -rn needle bindir/bin5.dat bindir/bin4.dat",
            "grep -a needle bindir/bin5.dat bindir/bin4.dat",
            "grep -rn needle bindir/bin5.dat bindir/bin6.dat",
            // ── Symlinks (BSD rules) ──
            "grep -r x plain/dirlink",
            "grep -r x plain/dirlink/",
            "grep -r foo filelink",
            "grep foo filelink a.txt",
            "grep -r x plain/dirlink2",
            // ── --include/--exclude/--exclude-dir (BSD last-match-wins) ──
            "grep -r --exclude='c.txt' x plain",
            "grep -r --exclude='plain/d1/*.txt' x plain",
            "grep -r --exclude-dir='d1' x plain",
            "grep -r --include='*.txt' x plain",
            "grep -r --include='*.txt' --exclude='c.txt' x plain",
            "grep -r --exclude='c.txt' --include='*.txt' x plain",
            "grep --exclude='b.txt' x a.txt b.txt",
            "grep --include='a.txt' x a.txt b.txt",
            "grep -r --include='plain/d1/*' --exclude='*.txt' x plain",
            "grep -r --exclude='*.txt' --include='plain/d1/*' x plain",
            // ── cd chains, ~ operands, globs ──
            "cd sub && grep -r needle .",
            "cd sub && grep -rn needle .",
            "grep -rn needle ~/htree",
            "grep -rn needle ~/htree/sub",
            "grep -rn needle sub",
            "grep -rn needle plain/*.txt",
            "grep -rn needle plain/d1/*.txt",
            "grep -rn needle ~/htree/*.txt",
            // ── -- and option permutation ──
            "grep -n -- x a.txt",
            "grep x -n a.txt",
            "grep -e x a.txt b.txt",
            "grep -e x -e y a.txt b.txt",
            "grep -y x a.txt",
            "grep -u -n x a.txt",
            "grep -r x -n plain",
            "grep -rn -- -x dash.txt",
            "grep -e x -- -x1", // dash-prefixed operand after `--`
            // ── head pipeline (grep part parity) ──
            "grep -rn x plain | head -5",
            "grep -rn x plain | head",
        ];

        let mut failures = Vec::new();
        for row in rows {
            let result = std::panic::catch_unwind(|| assert_parity(row, &ws, &home));
            if let Err(payload) = result {
                let msg = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(ToString::to_string))
                    .unwrap_or_else(|| "unknown panic".into());
                failures.push(format!("{row}: {msg}"));
            }
        }
        assert!(
            failures.is_empty(),
            "parity failures:\n  {}",
            failures.join("\n  ")
        );
    }

    #[test]
    fn fallback_triggers() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let ws = tmp.path().join("ws");
        let home = tmp.path().join("home");
        fs::create_dir_all(&ws).expect("ws");
        fs::create_dir_all(&home).expect("home");
        build_fixture(&ws, &home);

        let rows: &[&str] = &[
            "grep x",                                  // stdin
            "echo hi | grep x",                        // piped stdin
            "grep x -",                                // - operand
            "grep -q x a.txt",                         // -q
            "grep -L x a.txt",                         // -L
            "grep -P x a.txt",                         // -P
            "grep -z x a.txt",                         // -z
            "grep -U x a.txt",                         // -U
            "grep -S -r x plain",                      // -S
            "grep -d skip x plain",                    // -d
            "grep -b x a.txt",                         // -b
            "grep -f pat.txt a.txt",                   // -f
            "grep -m0 x a.txt",                        // -m0
            "grep -o -m2 x a.txt",                     // -m+-o
            "grep -o -A1 x a.txt",                     // -o+ctx
            "grep -v -A1 x a.txt",                     // -v+ctx
            "grep -w 'foo.' a.txt",                    // -w edge
            "grep -w '' a.txt",                        // -w empty
            "grep '\\(foo\\)\\1' br.txt",              // backreference
            "grep 'a\\{,2\\}' br.txt",                 // bad interval
            "grep 'a\\s' a.txt",                       // \s shorthand
            "grep '\\<foo\\>' br.txt",                 // \< \>
            "grep -e '' -e foo a.txt",                 // empty alternation
            "grep $VAR a.txt b.txt",                   // unexpanded variable
            "grep -o 'err\\|error' a.txt b.txt",       // -o + alternation (leftmost-longest)
            "grep -o 'x\\(a\\|ab\\)' br.txt a.txt",    // -o + nested alternation
            "grep x *.rs",                             // unexpandable glob
            "grep x a.txt",                            // single file (perf gate)
            "grep -c x a.txt",                         // single file, -c (perf gate)
            "grep -l x a.txt",                         // single file, -l (perf gate)
            "grep -r x plain/e.txt",                   // single file, recursive
            "cat <<EOF\nfoo\nEOF\ngrep x a.txt b.txt", // heredoc feeds a non-grep member
            "git grep x",                              // git grep
            "if grep x a.txt; then echo hi; fi",       // compound
            "for x in a; do grep x a.txt; done",       // compound
            "case $x in a) grep x a.txt;; esac",       // compound (case arm)
            "( grep x a.txt b.txt )",                  // subshell group
            "{ grep x a.txt; }",                       // brace group
            "(cd sub && grep x a.txt b.txt)",          // subshell cd + group
            "xargs grep x",                            // indirect
            "sh -c 'grep x a.txt'",                    // indirect
            "cd $HOME && grep x a.txt",                // cd untrackable
            "cd - && grep x a.txt",                    // cd $OLDPWD
            "grep x a.txt | head -1 | wc -l",          // pipeline shape
            "! grep x a.txt",                          // negation
            "sudo grep x a.txt",                       // env prefix
            "grep -w 'foo\\|bar' a.txt b.txt",         // -w + alternation (word_safe)
            "echo $(echo \\) ; grep x a.txt", // unterminated $(...) — escape-aware span stays open
            "echo hello && echo world",       // no grep at all
        ];
        for row in rows {
            assert_falls_back(row, &ws, &home);
        }
    }

    #[test]
    fn substitution_escape_resegments() {
        // Escape-aware substitution scans (consume_substitution) change
        // segmentation: an escaped backtick no longer truncates a backtick
        // span, so the trailing grep is a real separate member and gets served
        // (the old scan swallowed it into an unterminated span → fallback).
        // The serve→fallback direction of the same fix is a fallback_triggers
        // row (`echo $(echo \) ; grep x a.txt`).
        let tmp = tempfile::tempdir().expect("tempdir");
        let ws = tmp.path().join("ws");
        let home = tmp.path().join("home");
        fs::create_dir_all(&ws).expect("ws");
        fs::create_dir_all(&home).expect("home");
        build_fixture(&ws, &home);
        analyze_command("echo `a\\`b` ; grep x a.txt b.txt", &ws, &home, true)
            .expect("escaped backtick: trailing grep is a separate served member");
    }

    #[test]
    fn exclusion_delta_is_rg_default() {
        // The one approved behavioral delta: recursive walks skip
        // hidden/gitignored content; explicit file operands always searched.
        let tmp = tempfile::tempdir().expect("tempdir");
        let ws = tmp.path().join("ws");
        let home = tmp.path().join("home");
        fs::create_dir_all(&ws).expect("ws");
        fs::create_dir_all(&home).expect("home");
        build_fixture(&ws, &home);

        let (specs, _) =
            analyze_command("grep -r x ign", &ws, &home, true).expect("ign walk servable");
        let (eout, _, ecode) = engine_run(&specs[0]);
        let text = String::from_utf8_lossy(&eout).to_string();
        assert_eq!(ecode, 0);
        assert!(text.contains("ign/visible.txt"), "visible searched: {text}");
        assert!(!text.contains(".hidden.txt"), "hidden skipped: {text}");
        assert!(!text.contains("skip.log"), "gitignored skipped: {text}");

        // Explicit file operands bypass the exclusion filters entirely.
        let (specs, _) =
            analyze_command("grep -r x ign/.hidden.txt ign/skip.log", &ws, &home, true)
                .expect("explicit operands servable");
        let (eout, _, ecode) = engine_run(&specs[0]);
        assert_eq!(ecode, 0);
        let text = String::from_utf8_lossy(&eout).to_string();
        assert!(text.contains("ign/.hidden.txt"), "explicit hidden: {text}");
        assert!(text.contains("ign/skip.log"), "explicit gitignored: {text}");
    }

    #[test]
    fn engine_falls_back_on_untranslatable_patterns() {
        // The engine must never silently broaden/narrow a match set: patterns
        // that cannot be translated or compiled fall back instead.
        let tmp = tempfile::tempdir().expect("tempdir");
        let ws = tmp.path().join("ws");
        let home = tmp.path().join("home");
        fs::create_dir_all(&ws).expect("ws");
        fs::create_dir_all(&home).expect("home");
        build_fixture(&ws, &home);
        let rows: &[&str] = &[
            "grep '\\(' a.txt",
            "grep -E 'a(' a.txt",
            "grep 'a\\}' a.txt",
            "grep 'a{' a.txt", // literal { in BRE — servable, not fallback
        ];
        for row in &rows[..3] {
            assert_falls_back(row, &ws, &home);
        }
        assert_parity(rows[3], &ws, &home);
    }
}
