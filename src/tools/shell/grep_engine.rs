//! Transparent grep/egrep/fgrep interception for the shell tool, in BOTH
//! read-only and full modes (the engine itself is inherently read-only). The
//! read-only branch re-validates the rewrite through `shell::readonly` before
//! running it, and the rewrite's verb is a quoted absolute executable path
//! (single-quoted on unix, double-quoted on Windows), so the guard must accept
//! that spelling for the mode to serve anything at all.
//!
//! The shell guard passes commands to a real shell, so the analysis is per
//! platform and reads the same single value the spawn side does
//! ([`SHELL_PLATFORM`]): `sh -c` on unix, `cmd.exe /C` on Windows, whose
//! segmenting, quoting, verb dispatch and `cd` grammar the `windows`
//! submodule owns. Every platform *decision* reads that one runtime value —
//! passed through the analysis as a parameter (so a host test lane can drive
//! either platform) and read as the constant on the engine side — and every
//! verb list is consulted through the platform's own key
//! ([`list_key`]/[`windows::verb_key`]), so the interpreter's case- and
//! `.exe`-insensitive dispatch applies wherever a verb is classified; only OS
//! APIs (`exec`, `creation_flags`, the SIGPIPE disposition) are `#[cfg]`-gated.
//! Grep-family invocations that can be served are rewritten to
//! a hidden `__grep-engine` subcommand of the current binary (dispatched
//! before instance-lock acquisition in `main()`), which runs the ripgrep
//! substrate (grep-regex/grep-searcher + the ignore crate for rg-default
//! recursive-walk exclusions). Substitution is per-segment: an unservable grep
//! (single-file perf gate, unsupported flag, compound/nested shape, …) is kept
//! verbatim while a servable sibling grep elsewhere in the command is still
//! served. Anything not provably safe executes the original command unchanged
//! (fallback).
//!
//! That per-segment tolerance is unix-only. On Windows an unserved member would
//! be left to run a `grep` the platform does not have, and the interpreter's
//! "not recognized" failure is the very status a "no match" produces elsewhere,
//! so a command carrying a grep-family invocation is either fully served by the
//! engine or refused as an explicit failure ([`unserved_failure`],
//! [`GrepServe::refusal`]), except for the shapes listed below, which are left
//! to the platform as written.
//! That decision is fail-closed on *words*, not on meanings: a segment that
//! merely carries a grep-family word in a followed program's argument list
//! refuses too ([`GREP_INTRODUCERS`]) — unless that program owns the search
//! itself ([`SEARCH_OWNING_VERBS`]) — and a line the cmd model cannot read at
//! all refuses when a command-position search can still be seen in it
//! ([`unreadable_line_search`]); both over-refusals are deliberate, and the
//! loud failure names the cause. The shapes left to the platform carry no grep
//! member in command position: a search another program owns
//! ([`SEARCH_OWNING_VERBS`] — `git grep …`, `docker … grep …`,
//! `ssh host grep …` — read as a skipped member, never claimed), one in the
//! argument list of a program no list names (`foo grep x`), a command word that
//! is no grep verb — a path-spelled program (`.\grep x f.txt`), a grep word
//! glued to a leading `{` (`{grep x f.txt`; `{grep` is no shell verb on either
//! platform, and its `(` spelling is refused as an unreadable line instead), or
//! a redirect glued to the verb (`grep>x f.txt`) — and the
//! `find … -exec grep … \;` spelling, whose unquoted `;` stops the cmd reading
//! before any member exists. Those run the platform's own program and report
//! that program's outcome — an error when it is missing or reads the spelling
//! differently, never the engine's empty match set. The engine itself
//! re-validates and hands the member back to the real grep on any runtime doubt
//! — in place via `exec` on unix; on Windows, where there is no system `grep` to
//! replace ourselves with, it reports the reason on stderr and exits with the
//! sentinel code the parent refuses the call on, alongside
//! [`ENGINE_REFUSAL_MARKER`] — the marker keeps that refusal recognisable when a
//! pipeline tail masks the exit status.
//!
//! The engine enables the fast matcher (SIMD literal prefilter) and uses a
//! parallel recursive walk, so cross-file output ordering may differ from the
//! host BSD grep (in-file ordering is stable). The approved behavioral deltas:
//! recursive walks skip hidden/gitignored content (rg defaults; a served tail
//! like `grep -rn … | wc -l` sees that filtered stream) and `-o` + alternation
//! stays fail-closed (a match-set/span difference, not an ordering one).
//!
//! On unix the segment and word reading is the shell's own: an unquoted,
//! unescaped `<`/`>` ends the word before it and starts a redirection (the
//! digits before the operator belong to it only when everything before it is
//! digits), `&>` is one operator with the word before it kept, a bare `&`
//! backgrounds the member, an unquoted `#` opening a word takes the rest of
//! its line out of the command, and `>(…)`/`<(…)` is process substitution,
//! never a redirect (that member stays on the real `grep`). Each recognized
//! operator's raw spelling rides the rewrite verbatim, so the surrounding `sh`
//! performs the write or the backgrounding. Windows refuses the glued spelling
//! instead ([`windows::has_glued_redirect`]).
//!
//! Parity target is the host BSD grep under the shell tool's pinned
//! `LC_ALL=C.UTF-8`. The macOS-gated differential matrix is the authoritative
//! parity gate; recursive-walk rows compare as sorted line-sets (parallel
//! ordering), everything else byte-exact. There is no such gate on Windows
//! (no host, no system `grep`): that lane's reading is argued in `windows` and
//! its one host-testable piece, [`windows::fnmatch`], is pinned differentially
//! against this host's own `fnmatch`. Grep decisions are recorded in the
//! dedicated `grep_telemetry` table (logs DB), not the general log stream.

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use grep_matcher::Matcher;
use serde::{Deserialize, Serialize};

use crate::tools::path::shell_quote;
use crate::tools::shell::SHELL_PIPE_READ_CAP;
use crate::tools::shell::scan::CdScan;
use crate::tools::shell::scan::strip_heredoc_bodies;
use crate::util::UnwrapPoison;
use crate::util::is_word_char;

use super::{SHELL_PLATFORM, ShellPlatform};

mod windows;

// ── Protocol constants ────────────────────────────────────────────────────

/// Hidden subcommand name dispatched before lock acquisition (like `debug`).
const ENGINE_VERB: &str = "__grep-engine";
/// Spec JSON protocol version; mismatches make the engine fall back.
/// Bumped for the `stdin`/`report_stream_bytes` fields (producer-first stdin-fed
/// serving): a spec from a swapped binary is caught here (both directions)
/// and execs the real grep via the fallback argv, which carries the new
/// surface. The version check runs before any stdin read, so the exec'd grep
/// reads the producer's pipe authentically.
const PROTOCOL_VERSION: u32 = 5;
/// NUL-detection window for binary files (FreeBSD grep reads 32 KiB).
const BINARY_WINDOW: usize = 32 * 1024;
/// Engine self-cap on written output; the shell pipe reader caps at the same.
const OUTPUT_CAP: usize = SHELL_PIPE_READ_CAP;
/// Sentinel exit code: engine could not serve and could not exec grep either.
/// Unix: the parent re-runs the original command on this code. Windows: the
/// parent refuses the call — there is no system `grep` to re-run there — and
/// reports the engine's own stderr reason to the agent.
pub(super) const ENGINE_FAILED_EXIT: i32 = 3;
/// Exit code for a member whose downstream pipe closed. On unix the default
/// SIGPIPE disposition produces it (128 + SIGPIPE) and this constant is unused;
/// on Windows there is no SIGPIPE, so the equivalent `BrokenPipe` write error
/// exits with the same code — a `grep … | head` tail must stop the walk rather
/// than let it scan the whole tree.
#[cfg_attr(unix, expect(dead_code, reason = "the unix close path is SIGPIPE"))]
const BROKEN_PIPE_EXIT: i32 = 128 + 13;
/// Stderr signature of a stale self-update binary (one lacking the hidden
/// subcommand) running full main() and dying at instance-lock acquisition —
/// its exit 1 is a legitimate grep no-match code, so the parent treats this
/// message like the sentinel exit instead: unix re-runs the original command,
/// Windows refuses the call. Mirrors `self_update::acquire_lock`'s error text.
pub(super) const STALE_BINARY_LOCK_MSG: &str = "Another instance of mahbot is already running";
/// Engine stderr marker on a run that could not serve its member. The parent
/// refuses a Windows call on this marker even when the sentinel exit status was
/// masked by a pipeline tail (`grep … | tail -1` carries the tail's status),
/// which would otherwise reach the agent looking like an empty match set. It is
/// emitted on its own line — the parent recognises it by line equality, never as
/// a substring, so a served run whose matched line merely echoes the token is
/// not mistaken for a refusal — so the parent's cause extraction still reads the
/// human reason from the lines around it. Like the stream-size marker it rides
/// stderr, so three member-side spellings lose it: merging stderr into the
/// stream, redirecting fd 2 to a file (`grep … 2>err.txt | tail -1`), and a
/// preceding member flooding the parent's first-256 KB stderr cap
/// (`SHELL_PIPE_READ_CAP`) before the engine starts. Where the file swallows it
/// and the tail's exit status masks the sentinel, an unserved search can still
/// surface looking like an empty result. Accepted residual. A fourth loss is
/// outside the engine: when cmd.exe cannot launch the program the rewrite names
/// (this binary removed or quarantined after the probe accepted it), the run
/// carries the interpreter's own "not recognized" status and no marker at all —
/// also accepted, since the probe has just validated that path.
pub(super) const ENGINE_REFUSAL_MARKER: &str = "__mahbot_grep_engine_unserved__";
/// Engine stderr marker carrying the byte count consumed from a stdin-fed
/// stream (best-effort: -m/-l early stops report the consumed prefix,
/// SIGPIPE-killed chains may flush nothing, and member-side/shell-level
/// stderr merges suppress it). The shell tool strips the marker line from the
/// agent-visible stderr and logs the count (per-call stream bytes are
/// recorded nowhere else).
const STREAM_SIZE_MARKER: &str = "__mahbot_stream_bytes__";
/// Specs larger than this are not served on unix: the payload rides a single
/// argv entry there (argv-size hygiene).
const MAX_SPEC_JSON: usize = 64 * 1024;
/// The Windows limit: the payload rides a scratch file, not the command line,
/// so only a pathological expansion (a glob matching tens of thousands of
/// operands) hits it — the argv bound above would refuse ordinary `grep -rn
/// <pat> *` searches, whose expansion is large but perfectly servable.
const MAX_SPEC_FILE: usize = 4 * 1024 * 1024;
/// Searcher line-buffer cap; exceeding it is a grep-style error (exit 2).
const HEAP_LIMIT: usize = 512 * 1024 * 1024;

// ── Spec: the JSON protocol between parent and engine ─────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Original post-expansion argv for the exec-in-place fallback (unix only:
    /// the Windows engine has no `grep` to replace itself with and exits on the
    /// sentinel instead).
    fallback: Vec<String>,
    /// The member feeds a pipeline tail: its stdout must not be capped, so
    /// downstream members see the full stream (byte-identity with grep).
    #[serde(default)] // lenient: a swapped-binary spec reaches the version check
    piped: bool,
    /// The member is a non-first pipeline member fed by a producer's stdout
    /// via stdin (no operands): read stdin instead of operands.
    #[serde(default)]
    stdin: bool,
    /// Emit the stream-size marker to stderr (stdin serves without a
    /// member-side or shell-level stderr merge; the shell tool strips and
    /// logs it).
    #[serde(default)]
    report_stream_bytes: bool,
}

/// Output cap handed to the writer: unpiped serves self-cap at OUTPUT_CAP,
/// piped members are capped by the shell pipe reader instead (byte-identity
/// with grep).
fn output_limit(spec: &EngineSpec) -> Option<usize> {
    if spec.piped { None } else { Some(OUTPUT_CAP) }
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

/// One analyzed grep member's serve decision (for grep telemetry).
#[derive(Debug)]
pub(super) struct GrepOutcome {
    /// Whether this grep member was served by the engine.
    pub served: bool,
    /// The fallback reason when not served ('' when served).
    pub reason: String,
    /// Whether the grep was recursive (-r/-R).
    pub recursive: bool,
    /// Whether the grep sits in a pipeline.
    pub piped: bool,
    /// Number of resolved operands (files + directories).
    pub operand_count: usize,
    /// Normalized flag surface (e.g. "nivro...", single string).
    pub flags: String,
}

/// Result of analyzing a shell command for grep serving.
#[derive(Debug)]
pub(super) struct GrepServe {
    /// The rewritten command; `None` when no grep was served (original runs).
    pub rewritten: Option<String>,
    /// Per-grep-member decisions; empty when the command is not greppable.
    pub outcomes: Vec<GrepOutcome>,
    /// Scratch spec files the rewrite refers to (the Windows hand-off — the
    /// spec cannot ride on the cmd.exe command line); empty on unix. The
    /// caller owns them and removes them when the call finishes.
    pub spec_files: Vec<PathBuf>,
    /// The short cause of the search this platform could not serve, when the
    /// command does carry one. Always `None` on unix: an unserved member keeps
    /// falling back to the real `grep` there, so nothing is refused. On Windows
    /// the caller turns it into a tool error through [`unserved_failure`] — the
    /// single place every refusal's agent-facing message is rendered.
    pub refusal: Option<String>,
}

/// The agent-facing failure for a search the engine refuses to serve (the
/// Windows value of [`GrepServe::refusal`]). It becomes the tool error, so it
/// must be unmistakable for a result: it names the cause and says the search did
/// not run rather than matched nothing. Every refusal path — the serve
/// decision's own cause and the engine's reported one alike — renders its
/// message through here, so those two guarantees cannot drift between them.
///
/// No remedy is offered: which commands the engine can serve, and this
/// platform's quoting and `%` rules, are the shell description's business, and
/// one repeated here would be wrong for the causes no rewrite can avoid (an
/// unavailable engine, an unquotable hand-off path).
pub(super) fn unserved_failure(reason: &str) -> String {
    format!(
        "Search not served: the built-in grep engine cannot serve this command \
         on this platform, so the search did NOT run — this is not an empty \
         match set (cause: {reason})."
    )
}

/// Aggregate telemetry fields derived purely from the per-member outcomes
/// (shape + counts). The row-level `served`/`reason` depend on runtime facts
/// (whether the rewrite was actually applied, whether a sentinel re-run
/// replaced the engine) and are computed by the caller.
pub(super) struct GrepTelemetryShape {
    pub recursive: bool,
    pub piped: bool,
    pub operand_count: usize,
    pub flags: String,
    pub grep_count: usize,
    pub served_count: usize,
    pub skipped_count: usize,
}

impl GrepServe {
    /// A serve decision that keeps the original command running (no rewrite):
    /// every analyzed member stays as-is, and `refusal` carries the Windows
    /// failure when the platform refuses the command instead of running it.
    fn not_rewritten(outcomes: Vec<GrepOutcome>, refusal: Option<String>) -> Self {
        Self {
            rewritten: None,
            outcomes,
            spec_files: Vec::new(),
            refusal,
        }
    }

    /// Fold the per-member outcomes into the telemetry shape/count fields.
    /// `applied` gates served_count: when the rewrite wasn't actually applied
    /// (engine unavailable, spec too large, ReadOnly rejection, sentinel
    /// re-run) the engine served nothing, regardless of the per-member analysis.
    pub(super) fn telemetry_shape(&self, applied: bool) -> GrepTelemetryShape {
        let grep_count = self.outcomes.len();
        let served_count = if applied {
            self.outcomes.iter().filter(|o| o.served).count()
        } else {
            0
        };
        GrepTelemetryShape {
            recursive: self.outcomes.iter().any(|o| o.recursive),
            piped: self.outcomes.iter().any(|o| o.piped),
            operand_count: self.outcomes.iter().map(|o| o.operand_count).sum(),
            flags: self
                .outcomes
                .iter()
                .map(|o| o.flags.as_str())
                .filter(|s| !s.is_empty())
                .fold(String::new(), |mut acc, s| {
                    if !acc.is_empty() {
                        acc.push('|');
                    }
                    acc.push_str(s);
                    acc
                }),
            grep_count,
            served_count,
            skipped_count: grep_count - served_count,
        }
    }
}

/// Try to rewrite `command` into an engine-served equivalent, using the
/// running process's shell platform and engine probe. The engine is inherently
/// read-only (it preserves non-grep segments verbatim), so the caller runs it
/// in both ReadOnly and Full modes.
pub(super) fn try_serve_command(command: &str, workspace_root: &Path) -> GrepServe {
    serve_command(
        command,
        workspace_root,
        pinned_home().as_deref(),
        SHELL_PLATFORM,
        // The probe is deferred to the point where a member is servable (see
        // `serve_command`), so a shell call carrying no search never spawns it.
        engine_available,
    )
}

/// The Windows refusal for a serve decision that keeps the original command
/// running: the short cause of the search the command is known to carry, and
/// always `None` on unix, where an unserved member keeps falling back to the
/// real `grep`.
fn refusal(platform: ShellPlatform, cause: Option<String>) -> Option<String> {
    if platform == ShellPlatform::Windows {
        cause
    } else {
        None
    }
}

/// A serve decision demoted to the original command: no rewrite, every member
/// marked not-served with `reason` (the row's served field must match the
/// ACTUAL execution), and the platform's refusal — the command is known to
/// carry a search at every call site.
fn demoted(platform: ShellPlatform, mut outcomes: Vec<GrepOutcome>, reason: &str) -> GrepServe {
    all_not_served(&mut outcomes, reason);
    let refusal = refusal(platform, Some(reason.to_string()));
    GrepServe::not_rewritten(outcomes, refusal)
}

/// The single routing decision for one command: a served rewrite, or the
/// original command unchanged. The platform and the engine probe are explicit
/// parameters so both platforms' routing — and every demotion path — is
/// drivable from any host's unit-test lane.
///
/// The probe is a closure, not a result: answering it spawns the full binary,
/// so it is asked only once a complete analysis has left something to serve. A
/// command carrying no search never pays for it, and neither does one whose
/// members all turned out unservable.
///
/// On unix an unserved member keeps falling back to the real `grep`, so only a
/// produced rewrite ever changes what runs. On Windows a command carrying a
/// grep-family invocation is either fully served or refused
/// ([`GrepServe::refusal`]) — a half-served command would leave a member running
/// a program the platform has not got, which the module header explains must
/// never reach the shell.
///
/// `home` is the shell's pinned `$HOME`; `None` (no home directory resolved)
/// means nothing that could resolve `~` is served — except on Windows, where
/// nothing expands `~` at all and no analysis decision consults it, so the
/// analysis proceeds without one.
fn serve_command(
    command: &str,
    workspace_root: &Path,
    home: Option<&Path>,
    platform: ShellPlatform,
    engine_ready: impl Fn() -> bool,
) -> GrepServe {
    let windows = platform == ShellPlatform::Windows;
    let home = match home {
        Some(home) => home,
        // `resolve_operand`/`resolve_cd` never read `home` on Windows (their
        // `~` arms are unix-only), so an empty placeholder is never consulted —
        // the routing pin `windows_serves_without_a_home` holds this.
        None if windows => Path::new(""),
        None => return GrepServe::not_rewritten(Vec::new(), None),
    };
    // Single-file members are served on Windows: there is no host grep there to
    // be faster than, so the perf gate that keeps them on the real binary on
    // unix would only lose the serve.
    let allow_single = windows;
    let Analyzed {
        specs,
        shapes,
        segments,
        outcomes,
        unserved,
    } = match analyze_command(command, workspace_root, home, platform, allow_single) {
        Ok(analyzed) => analyzed,
        Err(fail) => {
            let AnalyzeFailure {
                reason,
                mut outcomes,
                unserved,
            } = fail;
            // The refusal cause: the member cause when a member exists, else —
            // Windows only — the tolerant command-position scan, which decides
            // whether a structural abort (`;`, heredoc, an unreadable line, a
            // trailing connector, all of which abort before any member exists)
            // even carried a search. A `NoGrep` line was read in full and has
            // no search, however the naive split reads it.
            let cause = unserved.or_else(|| {
                (windows
                    && !matches!(reason, Fallback::NoGrep)
                    && unreadable_line_search(command, platform))
                .then(|| reason.to_string())
            });
            // Pre-analysis structural failures carry no grep member — no
            // telemetry row is written. Otherwise the per-member outcomes are
            // preserved so an all-skipped command still records its shape; a
            // would-be serve discarded by a structural abort (untrackable `cd`)
            // is demoted so the row's served field matches the ACTUAL
            // (fallback) execution.
            if outcomes.is_empty() {
                return GrepServe::not_rewritten(Vec::new(), refusal(platform, cause));
            }
            tracing::debug!(command = command, %reason, "grep engine: fallback");
            if outcomes.iter().any(|o| o.served) {
                all_not_served(&mut outcomes, &reason.to_string());
            }
            return GrepServe::not_rewritten(outcomes, refusal(platform, cause));
        }
    };
    // A member the engine cannot serve is kept verbatim in the rewrite. On unix
    // that member is a real grep and the sibling serves still win; on Windows it
    // would run a program the platform does not have, so the whole command is
    // refused rather than half-served.
    if let Some(cause) = unserved {
        return demoted(platform, outcomes, &cause);
    }
    if !engine_ready() {
        return demoted(platform, outcomes, "engine unavailable");
    }
    // Serialized once per member: the same JSON is measured against the
    // platform's payload bound here and rendered into the fragment below.
    let jsons: Vec<String> = specs.iter().map(spec_json).collect();
    let limit = match platform {
        ShellPlatform::Unix => MAX_SPEC_JSON,
        ShellPlatform::Windows => MAX_SPEC_FILE,
    };
    if jsons.iter().any(|json| json.len() > limit) {
        return demoted(platform, outcomes, "spec exceeds payload limit");
    }
    // Only now — every analysis decision made, the whole command servable — is
    // the rewrite rendered, so the Windows hand-off's scratch files cannot be
    // created and then abandoned by a later demotion. The engine's own path is
    // part of that rendering: without it there is no rewrite, so a runtime doubt
    // demotes the command (unix runs the original, Windows refuses the call)
    // rather than panicking.
    let rendered = std::env::current_exe()
        .map(|exe| exe.to_string_lossy().into_owned())
        .map_err(|e| Fallback::Handoff(format!("no own executable path: {e}")))
        .and_then(|exe| join_rewritten(&segments, &jsons, platform, &exe));
    let (rewritten, spec_files) = match rendered {
        Ok(rendered) => rendered,
        Err(reason) => {
            tracing::debug!(command = command, %reason, "grep engine: fallback");
            return demoted(platform, outcomes, &reason.to_string());
        }
    };
    // Both served forms log at DEBUG — gate-relaxation volume telemetry is
    // too noisy for INFO (the dedicated telemetry table is the source of
    // truth). The stdin field marks producer-first stdin-fed serves.
    if shapes.is_empty() {
        tracing::debug!(
            command = command,
            greps = specs.len(),
            "grep engine: served"
        );
    } else {
        for (members, shape, stdin) in shapes {
            tracing::debug!(
                command = command,
                members,
                shape = %shape,
                stdin,
                "grep engine: served"
            );
        }
    }
    GrepServe {
        // `analyze_command` hands back `Err` when it served no member, so a
        // rewritten command here always carries at least one.
        rewritten: Some(rewritten),
        outcomes,
        spec_files,
        refusal: None,
    }
}

/// Best-effort `GrepOutcome` for a per-segment grep skip (compound/nested,
/// in-group, or an unservable member) that is kept verbatim in the rewrite.
fn skipped_grep_outcome(reason: String, seg: &str, piped: bool) -> GrepOutcome {
    let (recursive, operand_count, flags) = lightweight_scan(seg);
    GrepOutcome {
        served: false,
        reason,
        recursive,
        piped,
        operand_count,
        flags,
    }
}

/// Canonical telemetry flag order, used by `lightweight_scan` to render a
/// skipped member's flag string. `flags_surface` (served-member rendering)
/// independently follows the same relative order over the `GrepFlags`
/// fields; this sync is by convention only, with one deliberate
/// divergence: `flags_surface` folds before/after into `C` when both are
/// set, while `lightweight_scan` emits them as separate `A`/`B` letters.
const FLAG_ORDER: &[char] = &[
    'n', 'i', 'v', 'w', 'x', 'a', 'h', 'H', 's', 'r', 'o', 'z', 'c', 'l', 'm', 'C', 'A', 'B',
];

/// Telemetry scan of a command/segment for a skipped grep: detects
/// `-r`/`-R` (recursive), counts whitespace-separated non-flag words after
/// the verb (rough operand count), and collects the short-flag letters in
/// canonical order. Not a full parse — feeds telemetry only. Flag letters
/// keep their case so `-h`/`-H`, `-c`/`-C`, `-A`/`-B`/`-a` are
/// distinguishable, matching `flags_surface`'s encoding; `-L` is dropped
/// (absent from `FLAG_ORDER`).
fn lightweight_scan(command: &str) -> (bool, usize, String) {
    let mut recursive = false;
    let mut operand_count = 0usize;
    let mut flags: std::collections::BTreeSet<char> = std::collections::BTreeSet::new();
    let mut saw_pattern = false;
    for (i, word) in command.split_whitespace().enumerate() {
        if i == 0 {
            continue; // verb
        }
        if word.starts_with("--") {
            continue; // long option: not part of the short-flag surface
        }
        if let Some(cluster) = word.strip_prefix('-') {
            if cluster.is_empty() {
                // Bare `-` is a stdin operand, not a flag.
                if saw_pattern {
                    operand_count += 1;
                } else {
                    saw_pattern = true;
                }
                continue;
            }
            for c in cluster.chars().filter(char::is_ascii_alphabetic) {
                if c == 'r' || c == 'R' {
                    recursive = true;
                }
                // `-R` is parsed into `flags.r`, so record it as 'r' to match
                // `flags_surface` (FLAG_ORDER has no uppercase 'R').
                flags.insert(if c == 'R' { 'r' } else { c });
            }
            continue;
        }
        // Non-flag token: the first one is the positional pattern, the rest
        // are operands (best-effort — `-e` patterns over-count slightly).
        if saw_pattern {
            operand_count += 1;
        } else {
            saw_pattern = true;
        }
    }
    let flag_str = FLAG_ORDER
        .iter()
        .filter(|c| flags.contains(c))
        .copied()
        .collect();
    (recursive, operand_count, flag_str)
}

/// Rewrite `command` for a harness-supplied workspace/home, on the unix
/// platform the `sh`-driven e2e harness runs on. The production serve gate
/// (`engine_available`) is deliberately bypassed so the harness can drive the
/// rewrite directly; the analyser's own fallbacks still apply, and so does the
/// spec-size cap — enforced here, the way the production path enforces it in
/// [`serve_command`].
#[cfg(feature = "grep-engine-e2e")]
fn served_rewrite(command: &str, workspace_root: &Path, home: &Path) -> Option<String> {
    serve_command(
        command,
        workspace_root,
        Some(home),
        ShellPlatform::Unix,
        || true,
    )
    .rewritten
}

#[cfg(feature = "grep-engine-e2e")]
#[doc(hidden)]
#[must_use]
pub fn grep_engine_rewrite_for_test(
    command: &str,
    workspace_root: &Path,
    home: &Path,
) -> Option<String> {
    served_rewrite(command, workspace_root, home)
}

/// Whether a served spec exercises the parallel recursive directory walk
/// (`-r`/`-R` with at least one directory operand). Cross-file worker ordering
/// is non-deterministic for such rows, so parity comparisons relax to sorted
/// line-sets; in-file ordering stays stable. Only the macOS-gated parity matrix
/// and the e2e harness use this, so every other lane leaves it unused.
#[cfg_attr(
    not(any(all(test, target_os = "macos"), feature = "grep-engine-e2e")),
    expect(
        dead_code,
        reason = "used only by the macOS-gated parity matrix and the e2e harness"
    )
)]
fn spec_uses_parallel_walk(spec: &EngineSpec) -> bool {
    spec.flags.r
        && spec
            .operands
            .iter()
            .any(|op| std::fs::metadata(&op.resolved).is_ok_and(|m| m.is_dir()))
}

/// Whether the served grep member(s) for `command` exercise the parallel
/// recursive directory walk (the subprocess harness relaxes its byte-identical
/// diff to sorted-line comparison for those rows — cross-file worker ordering
/// is non-deterministic).
#[cfg(feature = "grep-engine-e2e")]
#[doc(hidden)]
#[must_use]
pub fn served_spec_walks_directory(command: &str, workspace_root: &Path, home: &Path) -> bool {
    let Ok(analyzed) = analyze_command(command, workspace_root, home, ShellPlatform::Unix, false)
    else {
        return false;
    };
    analyzed.specs.iter().any(spec_uses_parallel_walk)
}

/// Split `bytes` on `\n`/`\0` record terminators and sort the non-empty records
/// byte-wise. Ordering-insensitive output comparison for parallel-walk rows;
/// shared with the e2e bench so the differential gate uses the same matcher.
#[cfg(feature = "grep-engine-e2e")]
#[doc(hidden)]
#[must_use]
pub fn grep_sorted_lines(bytes: &[u8]) -> Vec<&[u8]> {
    let mut v: Vec<&[u8]> = bytes
        .split(|&b| b == b'\n' || b == b'\0')
        .filter(|l| !l.is_empty())
        .collect();
    v.sort();
    v
}

/// The shell tool's pinned `$HOME` — the child shell's synthetic baseline
/// (`UserDirs`), never the ambient daemon env.
fn pinned_home() -> Option<PathBuf> {
    directories::UserDirs::new().map(|d| d.home_dir().to_path_buf())
}

/// Serialize one spec for the hand-off. Never fails in practice (plain owned
/// data), and one function so the payload bound and the rendered fragment are
/// measured on the same string.
fn spec_json(spec: &EngineSpec) -> String {
    serde_json::to_string(spec).expect("spec serializes")
}

/// Remove engine stream-size marker line(s) from stderr; returns the last
/// reported byte count. Any line containing the marker token is removed —
/// the count parse only gates the returned value, so a non-marker line that
/// literally contains the token is lost too (accepted). Multi-pipeline
/// `;`-joined commands emit several markers and only the last is logged — an
/// accepted best-effort gap. The marker is engine-only telemetry and must
/// never reach the agent's stderr.
pub(super) fn strip_stream_size_marker(stderr: &mut Vec<u8>) -> Option<u64> {
    // Common path (file-operand serves never emit the marker): skip the copy.
    if !stderr
        .windows(STREAM_SIZE_MARKER.len())
        .any(|w| w == STREAM_SIZE_MARKER.as_bytes())
    {
        return None;
    }
    let mut last = None;
    let mut cleaned = Vec::with_capacity(stderr.len());
    let mut rest = &stderr[..];
    while let Some(pos) = rest
        .windows(STREAM_SIZE_MARKER.len())
        .position(|w| w == STREAM_SIZE_MARKER.as_bytes())
    {
        // Reuses locate_line (grep_searcher-mirroring line locator): the
        // parent-side strip must apply the same line-boundary semantics.
        let (line_start, line_end) = locate_line(rest, b'\n', pos);
        cleaned.extend_from_slice(&rest[..line_start]);
        if let Some(v) = std::str::from_utf8(&rest[pos + STREAM_SIZE_MARKER.len()..line_end])
            .ok()
            .and_then(|s| {
                s.trim()
                    .strip_prefix(':')
                    .and_then(|s| s.trim().parse::<u64>().ok())
            })
        {
            last = Some(v);
        }
        rest = &rest[line_end..];
    }
    cleaned.extend_from_slice(rest);
    *stderr = cleaned;
    last
}

/// Process-lifetime mark that the engine has answered a probe: only THAT is
/// cached. A probe spawns the full binary, so a success must not be paid for
/// again — but a failure must be able to heal, because on Windows the result is
/// load-bearing (no system `grep` to fall back to) and a one-shot failure (a
/// spawn refused under resource pressure, the binary caught mid-swap during a
/// self-update) would otherwise turn every later search into a hard refusal
/// until the daemon restarts. A persistently stale binary then pays one probe
/// per call that has something to serve — [`serve_command`] asks only after the
/// analysis produced a servable member — and the engine's own version check and
/// the parent's lock-message handling are that case's other safety nets.
static ENGINE_AVAILABLE: std::sync::OnceLock<()> = std::sync::OnceLock::new();

fn engine_available() -> bool {
    if ENGINE_AVAILABLE.get().is_some() {
        return true;
    }
    let available = probe_engine();
    if available {
        // Only success is cached — see the mark's doc.
        let _ = ENGINE_AVAILABLE.set(());
    }
    available
}

fn probe_engine() -> bool {
    let Some(exe) = std::env::current_exe().ok() else {
        return false;
    };
    let mut probe = std::process::Command::new(exe);
    probe
        .arg(ENGINE_VERB)
        .arg("--probe")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    // No console window for the probe child (the daemon also has none).
    // `creation_flags` is a Windows-only API, so this — unlike every platform
    // *decision* in this module tree — has to be a `cfg`.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        probe.creation_flags(super::CREATE_NO_WINDOW);
    }
    probe.status().is_ok_and(|s| s.success())
}

/// Render one served grep member into the shell fragment that runs the engine:
/// the current executable, the engine verb and the spec, plus the member's own
/// redirect tokens kept verbatim.
///
/// The spec reaches the engine by argv on unix (`sh -c` puts no meaningful
/// payload limit on it) and by scratch file on Windows ([`windows::write_spec_file`]):
/// cmd.exe caps its command line at 8191 characters and re-parses it before the
/// program sees the argv, so the JSON — which carries the agent's pattern and
/// operands — cannot ride on it. That scratch file is pushed onto `files`, the
/// caller's list of files to remove when the call finishes.
///
/// `exe` is this binary's path; it is an argument rather than a lookup so the
/// one machine-dependent part of a rewrite (an installation path cmd.exe reads
/// differently) is drivable from a test. `json` is the already-serialized spec —
/// serializing it here as well would repeat the work for a large operand set,
/// and the caller measures that same string against the payload bound.
fn render_served(
    json: &str,
    redirects: &[String],
    platform: ShellPlatform,
    exe: &str,
    files: &mut Vec<PathBuf>,
) -> Result<String, Fallback> {
    let mut fragment = String::new();
    match platform {
        ShellPlatform::Unix => {
            fragment.push_str(&shell_quote(exe));
            fragment.push(' ');
            fragment.push_str(ENGINE_VERB);
            fragment.push(' ');
            fragment.push_str(&shell_quote(json));
        }
        ShellPlatform::Windows => {
            // Quote and allocate before writing: a refused argument must not
            // leave a scratch file behind (see [`join_rewritten`]). The
            // allocated name goes onto `files` before the write: it is the
            // caller's to remove whether or not the write completes, or a
            // failed/partial write (ENOSPC/EIO) leaks the file it created.
            let quoted_exe = windows::cmd_quote(exe).map_err(Fallback::Handoff)?;
            let path = windows::spec_file_path();
            let quoted_path =
                windows::cmd_quote(&path.to_string_lossy()).map_err(Fallback::Handoff)?;
            files.push(path.clone());
            windows::write_spec_file(&path, json).map_err(Fallback::Handoff)?;
            fragment.push_str(&quoted_exe);
            fragment.push(' ');
            fragment.push_str(ENGINE_VERB);
            fragment.push(' ');
            fragment.push_str(windows::SPEC_FILE_FLAG);
            fragment.push(' ');
            fragment.push_str(&quoted_path);
        }
    }
    if !redirects.is_empty() {
        fragment.push(' ');
        fragment.push_str(&redirects.join(" "));
    }
    Ok(fragment)
}

/// Why a command was not served (telemetry + fail-closed decisions).
///
/// One enum for both platforms: the variants a platform cannot reach are simply
/// never constructed on it — `CmdSyntax` and `Expansion` come from the cmd.exe
/// model, `SingleFile` from the unix-only perf gate (Windows serves single-file
/// lookups, having no host grep to prefer).
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
    /// An unquoted `>(…)`/`<(…)`: process substitution is a word the shell
    /// builds from a command's output (or a pipe it feeds), never a
    /// redirection, and nothing about it resolves statically — the member
    /// stays on the real `grep`.
    ProcessSubstitution,
    SingleFile,
    CdUntrackable,
    StdinOperands,
    StdinRecursive,
    SegmentEmpty,
    /// A spelling of the command this module's cmd.exe reading cannot follow: a
    /// `^`, a command group, an unbalanced quote, an empty member — all of them
    /// spellings the interpreter itself reads — or a word whose glued operator
    /// this engine's token classifier keeps inside the word
    /// ([`windows::has_glued_redirect`]). The rendered cause names the engine as
    /// the reader that cannot follow it, never the interpreter's syntax.
    CmdSyntax(String),
    /// A word of the member carries two or more `%`, so cmd.exe may expand a
    /// `%…%` pair in it before any program sees its argv, and quoting cannot
    /// prevent that ([`windows::has_percent_expansion`] documents the per-word
    /// superset this test deliberately is — which is why the rendered cause
    /// states the character count rather than an expansion: `100%%` carries no
    /// pair).
    Expansion(String),
    /// The engine could not be handed what it needs: this binary's own path (any
    /// platform) or the scratch file the Windows transport carries the spec in.
    Handoff(String),
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
            Fallback::ProcessSubstitution => write!(f, "process substitution"),
            Fallback::SingleFile => write!(f, "single file"),
            Fallback::CdUntrackable => write!(f, "cd untrackable"),
            Fallback::StdinOperands => write!(f, "stdin with operands"),
            Fallback::StdinRecursive => write!(f, "stdin with -r"),
            Fallback::SegmentEmpty => write!(f, "empty command or pipeline member"),
            Fallback::CmdSyntax(s) => write!(f, "cmd.exe spelling the engine cannot read: {s}"),
            Fallback::Expansion(s) => write!(f, "two `%` in {s}"),
            Fallback::Handoff(s) => write!(f, "hand-off refused: {s}"),
        }
    }
}

/// A wholesale `analyze_command` failure: the command could not be served, so
/// the whole original runs (unix) or the call is refused (Windows). `reason`
/// names the cause; `outcomes` preserves the per-grep-member decisions collected
/// before the abort (empty for pre-analysis structural failures like
/// NoGrep/Heredoc/SegmentEmpty); `unserved` is the first unserved search member
/// (see [`Analyzed::unserved`]) — always `None` on unix.
#[derive(Debug)]
struct AnalyzeFailure {
    reason: Fallback,
    outcomes: Vec<GrepOutcome>,
    unserved: Option<String>,
}

impl From<Fallback> for AnalyzeFailure {
    fn from(reason: Fallback) -> Self {
        AnalyzeFailure {
            reason,
            outcomes: Vec::new(),
            unserved: None,
        }
    }
}

/// Demote every analyzed grep member to not-served with `reason`. The
/// telemetry served/skip fields must reflect the ACTUAL execution: when the
/// rewrite was abandoned (engine unavailable, spec too large, untrackable `cd`)
/// the engine served nothing, even if a member had been analyzed as servable.
/// Per-member shape fields are preserved for the analysis.
fn all_not_served(outcomes: &mut [GrepOutcome], reason: &str) {
    for o in outcomes {
        o.served = false;
        o.reason = reason.to_string();
    }
}

/// One member of the rewritten command: the original text kept verbatim, or a
/// served grep member (index into the analyzed specs) with the member-side
/// redirect tokens preserved verbatim in its spelling.
enum OutSegment {
    Verbatim(String),
    Served { spec: usize, redirects: Vec<String> },
}

/// Analyze output: one spec per served grep, the shape of every served
/// multi-member pipeline (members, verbs — grep-family normalized to "grep",
/// stdin-fed flag) for volume telemetry, the rewritten command's members with
/// their original connectors (rendered by [`join_rewritten`], so no scratch
/// file is created before the whole command is known to be servable), and the
/// per-grep serve decisions (telemetry).
struct Analyzed {
    specs: Vec<EngineSpec>,
    shapes: Vec<(usize, String, bool)>,
    segments: Vec<(OutSegment, String)>,
    outcomes: Vec<GrepOutcome>,
    /// Windows only (`None` on unix, where an unserved member simply falls back
    /// to the real `grep`): the cause of the FIRST member the engine cannot
    /// serve — see [`SEARCH_OWNING_VERBS`] for the one shape that is left
    /// unserved-but-allowed. A non-`None` value refuses the whole command
    /// ([`serve_command`]): a member kept verbatim in the rewrite would run a
    /// `grep` this platform does not have.
    unserved: Option<String>,
}

/// Record the first member this platform cannot serve. Windows only: there an
/// unserved member refuses the whole command (the platform has no `grep` to
/// fall back to), and the FIRST cause wins, naming the member the agent's
/// rewrite should start from. A no-op on unix, where the member simply falls
/// back to the real `grep`.
fn note_unserved(platform: ShellPlatform, unserved: &mut Option<String>, reason: &str) {
    if platform == ShellPlatform::Windows && unserved.is_none() {
        *unserved = Some(reason.to_string());
    }
}

/// Analyze a full shell command: segment it, track cds, serve every grep
/// member, keep everything else verbatim. Returns the [`Analyzed`] members or
/// the first fallback reason.
#[expect(clippy::too_many_lines)] // per-segment serve/skip decision loop
fn analyze_command(
    command: &str,
    workspace_root: &Path,
    home: &Path,
    platform: ShellPlatform,
    allow_single: bool,
) -> Result<Analyzed, AnalyzeFailure> {
    // A `;` outside quotes is not a command separator to cmd.exe (several of its
    // own commands read it as an argument delimiter), so the fragments around it
    // are not the commands the agent may have meant: fail closed rather than
    // serve a member whose reading the guard's own line scan disputes.
    if platform == ShellPlatform::Windows && windows::unquoted_semicolon(command) {
        return Err(Fallback::CmdSyntax("unquoted `;`".into()).into());
    }
    let stripped = strip_heredoc_bodies(command);
    if stripped != command {
        // strip_heredoc_bodies drops the `<<` marker, body and terminator (they
        // are stripped for read-only scanning). Rewriting around them would
        // leave a bare non-grep member reading inherited stdin — fail-closed.
        return Err(Fallback::Heredoc.into());
    }
    let segments = split_segments(&stripped, platform)?;
    if segments.is_empty() {
        return Err(Fallback::SegmentEmpty.into());
    }
    // No grep member anywhere: not an interception candidate. Before the shape
    // checks so non-grep pipelines (`cat f | head`) don't pollute the
    // pipeline-shape telemetry class with a mislabeled reason.
    if !segments.iter().any(|(seg, _)| {
        let verb = first_word(seg);
        is_grep_verb(verb, platform)
            || segment_contains_grep(seg, &list_key(verb, platform), platform)
    }) {
        return Err(Fallback::NoGrep.into());
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

    let mut cwd = canonical_or_lexical(workspace_root, platform);
    let mut rewritten: Vec<(OutSegment, String)> = Vec::new();
    let mut specs: Vec<EngineSpec> = Vec::new();
    let mut shapes: Vec<(usize, String, bool)> = Vec::new();
    // A shell-level `exec` stderr redirect seen so far (greps after it cannot
    // report the stream-size marker).
    let mut exec_redirects_stderr = false;
    // Members after a served grep are preserved verbatim — real tools on a
    // byte-identical uncapped stream, never analyzed (only the first grep in
    // a pipeline is served; later greps stay real).
    let mut tail_preserved = vec![false; n];
    // Per-grep serve decisions (telemetry), collected for every analyzed grep
    // member (served or skipped). The first per-segment skip reason is kept
    // so an all-skipped command still falls back wholesale with a cause.
    let mut outcomes: Vec<GrepOutcome> = Vec::new();
    let mut first_skip_reason: Option<Fallback> = None;
    // Windows only: the cause of the first member the engine cannot serve (see
    // [`Analyzed::unserved`]). `unix` never sets it: an unserved member keeps
    // falling back to the real `grep` there.
    let mut unserved: Option<String> = None;
    // Group depth across segments (unquoted `(`/`{` openers minus closers):
    // greps inside an open compound group are never served, even when the
    // segment splitter separated them from the group opener (e.g. `( cd d &&
    // grep … )` splits the grep out of the `(` segment).
    let mut group_depth = 0isize;

    for (idx, (seg, conn)) in segments.iter().enumerate() {
        let in_group = group_depth > 0;
        let delta = group_delta(seg);
        // Preserved tails bypass all analysis (cd/grep/compound checks
        // included): second/third greps and grep introducers (xargs grep,
        // sh -c, cd) in tail positions are real tools on that stream.
        if tail_preserved[idx] {
            // A second grep in the pipeline (`grep … | grep …`) is such a
            // preserved member: it consumes the engine's stream through the
            // pipeline. Windows has no program to run there, so the command is
            // refused rather than left with a member nothing can execute.
            if grep_family(first_word(seg), platform).is_some() {
                note_unserved(platform, &mut unserved, &Fallback::NestedGrep.to_string());
            }
            rewritten.push((OutSegment::Verbatim(seg.clone()), conn.clone()));
            group_depth = (group_depth + delta).max(0);
            continue;
        }
        let verb = first_word(seg);
        // Every verb list below is read through the platform's own key, so a
        // verb classifies identically however the interpreter spells it.
        let key = list_key(verb, platform);
        if is_cd_segment(verb, platform) {
            if pstart[idx] != pend[idx] {
                return Err(AnalyzeFailure {
                    reason: Fallback::CdUntrackable,
                    outcomes,
                    unserved,
                });
            }
            let new_cwd = match resolve_cd(seg, &cwd, home, platform) {
                Ok(c) => c,
                Err(reason) => {
                    return Err(AnalyzeFailure {
                        reason,
                        outcomes,
                        unserved,
                    });
                }
            };
            cwd = new_cwd;
            rewritten.push((OutSegment::Verbatim(seg.clone()), conn.clone()));
            group_depth = (group_depth + delta).max(0);
            continue;
        }
        if let Some(family) = grep_family(verb, platform) {
            // The first grep in a pipeline is served wherever it sits; a
            // non-first member is fed by the producer's stdout via stdin.
            // Later greps (grep-on-grep chains) are preserved verbatim.
            let ctx = PipelineCtx {
                piped: pstart[idx] != pend[idx],
                stdin_fed: pstart[idx] != idx,
                // A shell-level `exec 2>&1` before this point merges the
                // engine's stderr (the stream-size marker) into the tool's
                // captured stdout, where the parent's strip cannot reach it.
                marker_ok: !exec_redirects_stderr,
            };
            // Grep inside an open compound group: never served (the group's
            // opener segment was skipped; this member is inside the construct).
            if in_group {
                if ctx.piped {
                    tail_preserved[idx + 1..=pend[idx]].fill(true);
                }
                if first_skip_reason.is_none() {
                    first_skip_reason = Some(Fallback::NestedGrep);
                }
                note_unserved(platform, &mut unserved, &Fallback::NestedGrep.to_string());
                outcomes.push(skipped_grep_outcome(
                    Fallback::NestedGrep.to_string(),
                    seg,
                    ctx.piped,
                ));
                rewritten.push((OutSegment::Verbatim(seg.clone()), conn.clone()));
                group_depth = (group_depth + delta).max(0);
                continue;
            }
            match serve_one_grep(seg, family, &cwd, home, platform, allow_single, ctx) {
                Ok((spec, redirects)) => {
                    if ctx.piped {
                        tail_preserved[idx + 1..=pend[idx]].fill(true);
                        // Served-pipeline shape for volume telemetry; spans
                        // the full pipeline (producers, served grep, tail).
                        let verbs: Vec<String> = segments[pstart[idx]..=pend[idx]]
                            .iter()
                            .map(|(s, _)| {
                                let key = list_key(first_word(s), platform);
                                if is_grep_verb(&key, platform) {
                                    "grep".to_string()
                                } else {
                                    key
                                }
                            })
                            .collect();
                        shapes.push((verbs.len(), verbs.join("|"), spec.stdin));
                    }
                    outcomes.push(GrepOutcome {
                        served: true,
                        reason: String::new(),
                        recursive: spec.flags.r,
                        piped: ctx.piped,
                        operand_count: spec.operands.len(),
                        flags: flags_surface(&spec.flags),
                    });
                    rewritten.push((
                        OutSegment::Served {
                            spec: specs.len(),
                            redirects,
                        },
                        conn.clone(),
                    ));
                    specs.push(spec);
                }
                Err(e) => {
                    // Per-grep fallback: skip this segment (keep it verbatim),
                    // record the skip, and continue to the next — a servable
                    // sibling elsewhere in the command is still served (unix;
                    // on Windows the skipped member refuses the whole command —
                    // see `Analyzed::unserved`).
                    let reason = e.to_string();
                    if ctx.piped {
                        tail_preserved[idx + 1..=pend[idx]].fill(true);
                    }
                    if first_skip_reason.is_none() {
                        first_skip_reason = Some(e);
                    }
                    note_unserved(platform, &mut unserved, &reason);
                    outcomes.push(skipped_grep_outcome(reason, seg, ctx.piped));
                    rewritten.push((OutSegment::Verbatim(seg.clone()), conn.clone()));
                }
            }
            group_depth = (group_depth + delta).max(0);
            continue;
        }
        // Non-grep member: verbatim. In a pipeline it is a producer — the
        // first grep in that pipeline is served from its stdout. Indirect/
        // compound grep invocations (xargs grep, sh -c, sudo, for bodies) are
        // skipped per-segment (kept verbatim) rather than making the whole
        // command fall back; a served sibling grep elsewhere in the command is
        // still served. `git grep`/`docker run … grep` are the introducer's own
        // search, not one the engine could ever claim, so they are exempt from
        // the Windows refusal ([`SEARCH_OWNING_VERBS`]).
        let compound = is_compound_segment(seg, platform);
        let carries_grep = segment_contains_grep(seg, &key, platform);
        if compound || carries_grep {
            // A telemetry skip is recorded only when the segment actually
            // carries a grep member; bare construct keywords (`then`, `fi`,
            // `done`) and grep-less indirect prefixes must not inflate the
            // grep/skipped counts in the compound case.
            if carries_grep {
                if first_skip_reason.is_none() {
                    first_skip_reason = Some(Fallback::NestedGrep);
                }
                if !SEARCH_OWNING_VERBS.contains(&key.as_str()) {
                    note_unserved(platform, &mut unserved, &Fallback::NestedGrep.to_string());
                }
                outcomes.push(skipped_grep_outcome(
                    Fallback::NestedGrep.to_string(),
                    seg,
                    pstart[idx] != pend[idx],
                ));
            }
            rewritten.push((OutSegment::Verbatim(seg.clone()), conn.clone()));
            group_depth = (group_depth + delta).max(0);
            continue;
        }
        if key == "exec" && (seg.contains("2>") || seg.contains("&>")) {
            // `exec 2>&1`/`exec 2>…`/`exec &>…` moves the shell's stderr for
            // the rest of the command; fail-closed on the stream-size marker
            // for the greps that follow (it would leak into stdout or a file).
            // `exec 1>&2`/`exec 3>&2` don't move stderr off the capture and
            // are intentionally not matched; digit-prefixed dups (`exec 12>&1`)
            // match `2>` and suppress unnecessarily (telemetry loss only).
            // Fail-open escapes (a stray marker line in stdout): env-prefixed
            // `FOO=1 exec 2>&1` and escaped-verb `\exec 2>&1` both miss the
            // folded-key check.
            exec_redirects_stderr = true;
        }
        rewritten.push((OutSegment::Verbatim(seg.clone()), conn.clone()));
        group_depth = (group_depth + delta).max(0);
    }

    // Served iff at least one grep was served. If none was, the command falls
    // back wholesale (the original runs) with the first per-segment skip
    // reason naming the cause — a pure-skip command must not run the engine.
    // The collected per-member outcomes are preserved (the caller records the
    // shape; a would-be serve demoted by the whole-command abort is handled at
    // the call site).
    if specs.is_empty() {
        return Err(AnalyzeFailure {
            reason: first_skip_reason.unwrap_or(Fallback::NoGrep),
            outcomes,
            unserved,
        });
    }
    Ok(Analyzed {
        specs,
        shapes,
        segments: rewritten,
        outcomes,
        unserved,
    })
}

/// Join the rewritten members with their connectors into one command — verbatim
/// except that the Windows splitter's newline connector is re-emitted as `&` (see
/// the loop) — rendering every served member through [`render_served`]. The
/// rendered spec files are returned to the caller: on Windows they are the
/// hand-off's scratch files, created here — the first point at which the whole
/// command is known to be servable — and removed when the call finishes. A member
/// whose hand-off fails takes the files already written by its predecessors with
/// it: the command is demoted, so nobody else ever owns them.
///
/// `exe` is this binary's path — a parameter for the same reason
/// [`render_served`] takes it, so a test can drive the rendered command without
/// naming an installed executable.
fn join_rewritten(
    segments: &[(OutSegment, String)],
    jsons: &[String],
    platform: ShellPlatform,
    exe: &str,
) -> Result<(String, Vec<PathBuf>), Fallback> {
    let mut out = String::new();
    let mut files = Vec::new();
    for (i, (seg, conn)) in segments.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        match seg {
            OutSegment::Verbatim(text) => out.push_str(text),
            OutSegment::Served { spec, redirects } => {
                match render_served(&jsons[*spec], redirects, platform, exe, &mut files) {
                    Ok(fragment) => out.push_str(&fragment),
                    Err(reason) => {
                        discard_spec_files(&files);
                        return Err(reason);
                    }
                }
            }
        }
        if i + 1 < segments.len() {
            out.push(' ');
            // The Windows splitter's newline connector: cmd.exe reads a bare
            // newline as the same unconditional separator `&` spells, but the
            // rewrite rides one `/C "…"` argument and whether cmd splits there
            // is the one spelling this workspace cannot measure — getting it
            // wrong costs every command after the first line. `&` is
            // unambiguous, and the Windows splitter never emits an empty member
            // (blank lines are skipped), so the substitution cannot create a
            // syntax error.
            if platform == ShellPlatform::Windows && conn == "\n" {
                out.push('&');
            } else {
                out.push_str(conn);
            }
        } else if conn == "&" {
            // The last member's connector is normally a no-op (`;`, a newline)
            // and is dropped with its trailing position. A trailing `&`
            // backgrounds the member instead: dropping it would run the served
            // search in the foreground, under the member's own exit status
            // rather than the shell's. Windows never produces one — its
            // splitter fails closed on a trailing connector.
            out.push(' ');
            out.push_str(conn);
        }
    }
    Ok((out, files))
}

/// Remove scratch spec files, best-effort — the single home of that policy: the
/// engine calls it when a hand-off abandons a rewrite (a later member's
/// `render_served` failed), so a refused Windows search leaks none, and the
/// shell's `SpecFiles` guard calls it on drop, owning the lifecycle point. A
/// removal that fails is the temp cleaner's.
pub(super) fn discard_spec_files(files: &[PathBuf]) {
    for path in files {
        let _ = fs::remove_file(path);
    }
}

/// Split a command into (segment, following-connector) pairs, in the spelling
/// of `platform`'s own shell: `sh` connectors are `&&`, `||`, `;`, `|`, `|&`,
/// `&`, `\n` or `` (last); cmd.exe's are `&&`, `||`, `&`, `|`, `\n` or `` (last)
/// — `;` and `&` are not interchangeable between them, which is why the platform
/// is a parameter rather than a `cfg` branch. Quote- and substitution-aware
/// (per platform), heredoc bodies already stripped. Empty segments before a
/// connector (or a trailing `|`/`|&`/`||`/`&&`, and on Windows a bare `&`) are
/// shell syntax errors; the rewriting would silently drop them into a VALID
/// executed pipeline — fail-closed on the whole class. Blank lines (`\n`
/// between commands) are valid on both and stay allowed, as is a trailing `&`
/// (a backgrounded member).
fn split_segments(
    command: &str,
    platform: ShellPlatform,
) -> Result<Vec<(String, String)>, Fallback> {
    match platform {
        ShellPlatform::Unix => {
            super::segment_command(command, super::SegmentMode::Grep).ok_or(Fallback::SegmentEmpty)
        }
        // A line the cmd.exe model does not cover — a caret, a command group,
        // an unbalanced quote, an empty member before a connector — leaves the
        // whole line unread (the first three are spellings cmd.exe itself reads;
        // the empty member mirrors the unix segmenter's own fail-closed policy).
        ShellPlatform::Windows => windows::segment_command(command)
            .ok_or_else(|| Fallback::CmdSyntax("the whole line".into())),
    }
}

/// Net group-open delta contributed by one segment's unquoted `(`/`{` and
/// `)`/`}` delimiters. Quote- and escape-aware so pattern parens/braces
/// (`'a(b'`, `\(`) don't count; `${…}`/`$(…)`/`$((…))` balance to zero within
/// a segment (the `$`-expansion's own open/close pair cancels), so a real
/// group that spans segments (e.g. `( cd d && grep … )`) is counted correctly.
/// An unbalanced unquoted `(`/`{` inside a grep pattern or operand word (e.g.
/// `grep -E a(b file`, `grep -E x{2 file`) leaves the depth positive and
/// misclassifies a later top-level sibling grep as in-group — it is skipped
/// (fail-closed: the real grep runs, so the output is correct; only the serve
/// is wrongly declined). This errs on the skip side, so it is accepted.
fn group_delta(segment: &str) -> isize {
    let mut delta = 0isize;
    let mut in_single = false;
    let mut in_double = false;
    let mut chars = segment.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' && !in_single {
            chars.next();
            continue;
        }
        if !super::check_outside_quotes(c, &mut in_single, &mut in_double) {
            continue;
        }
        match c {
            '(' | '{' => delta += 1,
            ')' | '}' => delta -= 1,
            _ => {}
        }
    }
    delta
}

/// First whitespace-delimited word of a segment (raw, quote-preserving).
fn first_word(segment: &str) -> &str {
    segment.split_whitespace().next().unwrap_or("")
}

/// The grep family `verb` names on `platform`, or `None` when it names another
/// program: the literal `grep`/`egrep`/`fgrep` on unix, and on Windows whatever
/// cmd.exe would dispatch to — case-insensitively, through the executable
/// extension (see [`windows::grep_verb`]).
fn grep_family(verb: &str, platform: ShellPlatform) -> Option<&'static str> {
    match platform {
        ShellPlatform::Unix => match verb {
            "grep" => Some("grep"),
            "egrep" => Some("egrep"),
            "fgrep" => Some("fgrep"),
            _ => None,
        },
        ShellPlatform::Windows => windows::grep_verb(verb),
    }
}

fn is_grep_verb(verb: &str, platform: ShellPlatform) -> bool {
    grep_family(verb, platform).is_some()
}

/// The word a verb list is matched against on `platform`: cmd.exe dispatches
/// case-insensitively and through an `.exe` suffix ([`windows::verb_key`], which
/// reads one word), so `CMD /C`, `cmd /c` and `cmd.exe /c` must classify the
/// same way — a raw first word would let a spelling other than the lower-case
/// one slip past every list below (neither served nor refused, then reaching the
/// agent as the interpreter's own "not recognized" text). unix verbs are the
/// words themselves — a shell matches them case-sensitively — so nothing is
/// folded, except the grep family, which [`grep_family`] folds itself.
///
/// This is the list-matching form of that key: `sh`'s builtins are not programs,
/// so no `.exe`-folding applies on unix, and a word the platform would re-read
/// (an unquotable spelling) folds to the empty key, which matches no list entry.
fn list_key(verb: &str, platform: ShellPlatform) -> String {
    match platform {
        ShellPlatform::Unix => verb.to_string(),
        ShellPlatform::Windows => windows::verb_key(verb).unwrap_or_default(),
    }
}

/// True when the segment's first word names a cwd-changing builtin on
/// `platform`: `cd` alone on unix, and cmd.exe's whole family on Windows —
/// `cd`/`chdir` (the same builtin), `pushd` (changes to its target) and `popd`
/// (returns to the directory `pushd` remembered) — read through [`list_key`]
/// like every other verb list, which is the same `cd`/`chdir`/`popd`/`pushd`
/// grouping the read-only guard's own `INTERNAL_VERBS` uses. Tracking only `cd`
/// would leave the cwd stale for the rest of the family, and the engine's cwd
/// gate would refuse the served member at runtime; the unix reading stays
/// byte-exact.
fn is_cd_segment(verb: &str, platform: ShellPlatform) -> bool {
    match platform {
        ShellPlatform::Unix => verb == "cd",
        ShellPlatform::Windows => matches!(
            list_key(verb, platform).as_str(),
            "cd" | "chdir" | "popd" | "pushd"
        ),
    }
}

/// Command-introducer verbs whose argument list may contain a nested grep
/// invocation (compounds, indirect invocations), matched through [`list_key`]
/// so a spelling like `CMD /C` or `cmd.exe /c` is the same introducer. A
/// grep-family word in such a segment is kept verbatim — never served, since its
/// semantics are uncertain — and the segment is recorded as a telemetry skip; a
/// servable sibling grep elsewhere in the command is still served on unix,
/// while on Windows the carried word refuses the whole command (nothing there
/// could run the sibling the rewrite would leave behind). The other interpreters
/// (`cmd`, `powershell`, `pwsh`) are listed beside the unix shells for the same
/// reason: a grep inside another interpreter's command line is a nested search
/// the engine cannot serve, so on Windows it is refused rather than left to run
/// a `grep` the platform has not got. On unix those three names only add a
/// recorded row (`cmd /c grep x f.txt` leaves a `grep_telemetry` skip entry),
/// and neither the serve decision nor the command text changes there.
///
/// The match is the word, not the meaning: a segment that merely *carries* a
/// grep-family word is refused on Windows even when it is not a search at all
/// (`cmd /c del grep`). That fail-closed over-refusal is deliberate — the
/// alternative is letting a real search through — and the loud failure names
/// it; [`SEARCH_OWNING_VERBS`] is the one exemption. That list is the one to
/// revisit with this one when a verb's classification changes.
const GREP_INTRODUCERS: &[&str] = &[
    "if",
    "while",
    "until",
    "case",
    "for",
    "select",
    "then",
    "else",
    "elif",
    "do",
    "!",
    "time",
    "command",
    "builtin",
    "exec",
    "eval",
    "sudo",
    "env",
    "nice",
    "nohup",
    "xargs",
    "ssh",
    "sh",
    "bash",
    "zsh",
    "ksh",
    "dash",
    "csh",
    "tcsh",
    "fish",
    "cmd",
    "powershell",
    "pwsh",
    "find",
    "git",
    "docker",
    "kubectl",
    "podman",
];

/// Compound-construct keywords: a segment starting with one (or with a `(`/`{`
/// group opener) nests a compound construct, so no grep inside THAT segment is
/// ever served. A servable grep in a sibling segment outside the compound is
/// still served; only the nested member falls back.
const COMPOUND_KEYWORDS: &[&str] = &[
    "if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done", "case", "esac",
    "select",
];

/// True when a segment opens a compound construct (keyword prefix or a `(`/`{`
/// group opener) — such a command must not serve any grep. The keyword list is
/// read through [`list_key`], so cmd.exe's case-insensitive `IF`/`FOR` are the
/// same construct as `if`/`for` there (on unix the words stand as written).
fn is_compound_segment(segment: &str, platform: ShellPlatform) -> bool {
    let trimmed = segment.trim_start();
    match trimmed.chars().next() {
        Some('(' | '{') => true,
        Some(c) if is_word_char(c) => {
            COMPOUND_KEYWORDS.contains(&list_key(first_word(trimmed), platform).as_str())
        }
        _ => false,
    }
}

/// True when a non-grep segment could contain a grep invocation we would not
/// serve (compound constructs, indirect invocation). Conservative: any
/// grep-family word in an introducer segment, or an env-assignment verb. `verb`
/// is the segment's [`list_key`], so the lists are read the way the platform
/// dispatches.
///
/// The words are the platform's own: cmd.exe's tokenizer on Windows, which keeps
/// a quoted argument as the single word cmd.exe delivers. A quoted word's
/// delivered text is then read as the command line it is, because that is what a
/// wrapped interpreter receives — `cmd /c "grep -rn x ."` carries a search a
/// whitespace split sees only as the unbalanced token `"grep`, which would leave
/// the command neither served nor refused. unix keeps its own reading (a missed
/// telemetry row for `sh -c 'grep …'`, never a serve decision: unix has the real
/// `grep` to fall back to).
fn segment_contains_grep(segment: &str, verb: &str, platform: ShellPlatform) -> bool {
    let suspicious = GREP_INTRODUCERS.contains(&verb) || verb.contains('=');
    if !suspicious {
        return false;
    }
    match platform {
        ShellPlatform::Windows => {
            windows::tokenize(segment).is_some_and(|words| words_nest_grep(&words))
        }
        ShellPlatform::Unix => segment
            .split_whitespace()
            .any(|w| is_grep_verb(crate::tools::shell::scan::strip_quoted_word(w), platform)),
    }
}

/// [`segment_contains_grep`]'s word test for cmd.exe's own tokenizer output:
/// any delivered word — or the command line inside a quoted word, which is what
/// the wrapped interpreter receives — names the grep family. Recursion strips at
/// least one quote pair per round, so it terminates.
fn words_nest_grep(words: &[GrepWord]) -> bool {
    words.iter().any(|w| {
        is_grep_verb(&w.value, ShellPlatform::Windows)
            || (w.value != w.raw
                && windows::tokenize(&w.value).is_some_and(|inner| words_nest_grep(&inner)))
    })
}

/// Programs that own the search they run: `git grep`, `docker run … grep`,
/// `kubectl exec … grep`, `podman … grep` are that program's own search — the
/// engine never claims them and a `grep` word in their argument list is data,
/// not a member to serve. `ssh` is here for the same reason from the other side:
/// its argument list runs on another machine, so the local engine claiming the
/// search would be claiming the wrong host's files — and refusing it would
/// refuse a command this platform can perfectly well run.
///
/// A command whose ONLY grep word sits in one of these is left unserved but
/// allowed on Windows, where the platform refusal ([`Analyzed::unserved`])
/// applies to members the engine could otherwise serve. Matched through
/// [`list_key`], like every other verb list. `find` is deliberately NOT here:
/// `find … -exec grep …` would run the platform's own `grep`, which is what
/// Windows does not have — the `\;` spelling reaches the platform only because
/// its unquoted `;` aborts the analysis, and an abort is refused only when a
/// command-position search is visible in the line (this spelling shows none).
/// An accepted residual: the platform's own `find` and `grep` then run it their
/// way, and what they report is never this engine's empty match set.
/// [`GREP_INTRODUCERS`] is the list to revisit with this one when a verb's
/// classification changes.
const SEARCH_OWNING_VERBS: &[&str] = &["git", "docker", "kubectl", "podman", "ssh"];

/// True when `command` carries a grep-family invocation in command position,
/// by a deliberately tolerant scan: split on every separator cmd.exe acts on
/// unconditionally (`&`, `|`, newlines) plus, over-wide on purpose, `;` — which
/// cmd reads as ordinary text and the cmd model therefore refuses — with no
/// quote handling at all, then strip the leading whitespace and any group opener
/// (`(`/`{`) off each fragment and test its first word through the same verb
/// predicate the analyzer uses. The opener is stripped rather than split on, so
/// a `(grep …)` fragment is still read as the grep it is, while
/// `echo (grep is a tool)` — a grep word in argument position — stays the plain
/// echo it is.
///
/// It exists only to decide whether a line the cmd.exe model could not read AT
/// ALL (a caret, a group, an unbalanced quote, a `;`, a trailing connector)
/// should be refused on Windows: such a line aborts before any member exists, so
/// the analyzer's own reading cannot say. The heuristic errs toward refusing an
/// unreadable line, while leaving `git commit -m "grep fix"`, `echo grep` and a
/// search-owning `git grep x` alone (their command words are not grep-family).
///
/// Being quote-blind it can also lift a grep word out of a quoted string and
/// blame a line that carries no search (`echo "a; grep x" && cd /x` is refused
/// with the cause of the abort that actually happened, not of a search). Fail
/// closed, and accepted: a quote-aware scan would have to trust the quotes of a
/// line it exists because it could not read.
fn unreadable_line_search(command: &str, platform: ShellPlatform) -> bool {
    command.split(['&', '|', ';', '\n']).any(|fragment| {
        let fragment = fragment.trim_start_matches(['(', '{', ' ', '\t']);
        grep_family(first_word(fragment), platform).is_some()
    })
}

/// Resolve a literal cwd-family segment (see [`is_cd_segment`]) against the
/// tracked cwd. Returns the new canonical cwd. Only statically-resolvable
/// targets track; everything else falls back (fail-closed).
///
/// The grammar is the platform's own: `sh`'s option scan and `$HOME` target on
/// unix, cmd.exe's `/d`-only, nothing-changes-on-bare-`cd` reading (where `~`
/// is an ordinary directory name) on Windows. On Windows the family's other two
/// halves are fail-closed: `popd` returns to a directory only the pushed stack
/// knows, and a bare `pushd` pushes the cwd and changes to that drive's root —
/// neither is modelled, so both leave [`Fallback::CdUntrackable`] rather than a
/// wrong tracked cwd. `pushd <target>` is tracked exactly like `cd <target>`.
///
/// Two cmd.exe spellings stay outside this model, both refusing rather than
/// tracking a wrong directory: a glued switch (`cd/d C:\ws`, which the read-only
/// guard's own reader splits into `cd /d`), and a drive-relative target
/// (`cd C:foo`, relative to that drive's remembered directory, not to the
/// tracked cwd). Either way the served member's spec cwd and the engine's own
/// cwd diverge and the engine's cwd gate refuses the call — a loud refusal of a
/// command an agent could have meant, never a search of the wrong directory.
fn resolve_cd(
    segment: &str,
    cwd: &Path,
    home: &Path,
    platform: ShellPlatform,
) -> Result<PathBuf, Fallback> {
    if platform == ShellPlatform::Windows {
        // cmd.exe's own word reading, not a whitespace split: a quoted target
        // keeps its inner spaces, and the spelling (quotes included) is what
        // [`windows::cd_scan`] judges.
        let words = windows::tokenize(segment).ok_or(Fallback::CdUntrackable)?;
        let Some((verb, rest)) = words.split_first() else {
            return Err(Fallback::CdUntrackable);
        };
        let verb = list_key(&verb.raw, platform);
        let rest: Vec<&str> = rest.iter().map(|w| w.raw.as_str()).collect();
        // The stack is what `popd` reads and this model does not keep, so its
        // target (and therefore the cwd after it) is unknowable.
        if verb == "popd" {
            return Err(Fallback::CdUntrackable);
        }
        let (target, next) = match windows::cd_scan(&rest) {
            CdScan::Target(target, next) => (target, next),
            // A bare `cd`/`chdir` prints the cwd and changes nothing; a bare
            // `pushd` is not that reading, and its effect here is not modelled
            // — fail-closed.
            CdScan::Bare => {
                if verb == "pushd" {
                    return Err(Fallback::CdUntrackable);
                }
                return Ok(cwd.to_path_buf());
            }
            // A switch-shaped word (`/x`) or a spelling cmd reads differently
            // — fail-closed.
            CdScan::BadOption => return Err(Fallback::CdUntrackable),
        };
        if rest.get(next).is_some() {
            // Extra operands are shell-dependent — fail-closed.
            return Err(Fallback::CdUntrackable);
        }
        if windows::has_percent_expansion(target) {
            return Err(Fallback::CdUntrackable);
        }
        return Ok(canonical_or_lexical(
            &absolute_or_cwd(cwd, target, platform),
            platform,
        ));
    }
    let words: Vec<&str> = segment.split_whitespace().collect();
    let (target, next) = match super::scan::cd_target_after_options(&words, 1) {
        CdScan::Target(target, next) => (target, next),
        // Bare `cd` → $HOME.
        CdScan::Bare => return Ok(canonical_or_lexical(home, platform)),
        // Invalid options (`cd -e`) error at runtime — fail-closed.
        CdScan::BadOption => return Err(Fallback::CdUntrackable),
    };
    if words.get(next).is_some() {
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
    } else {
        absolute_or_cwd(cwd, target, platform)
    };
    Ok(canonical_or_lexical(&resolved, platform))
}

/// Resolve a shell path word: absolute paths pass through verbatim, anything
/// else is joined onto the tracked cwd. Each platform reads its own absolute
/// spelling (`/…` on unix; a drive root, a UNC share or a rooted separator on
/// Windows).
fn absolute_or_cwd(cwd: &Path, word: &str, platform: ShellPlatform) -> PathBuf {
    let absolute = match platform {
        ShellPlatform::Unix => word.starts_with('/'),
        ShellPlatform::Windows => windows::is_absolute_word(word),
    };
    if absolute {
        PathBuf::from(word)
    } else {
        cwd.join(word)
    }
}

/// Canonicalize an existing path, falling back to a lexically absolute form
/// (`std::path::absolute`, exact because every input is already absolute) and
/// then to the path as given. On Windows the verbatim (`\?\`) prefix
/// `fs::canonicalize` adds is stripped, so the tracked cwd, the spec's operand
/// paths and the displayed paths carry the spelling cmd.exe itself reports —
/// and the cwd gate then needs [`windows::same_directory`]'s
/// spelling-insensitive comparison rather than byte equality.
fn canonical_or_lexical(p: &Path, platform: ShellPlatform) -> PathBuf {
    let canonical = fs::canonicalize(p)
        .or_else(|_| std::path::absolute(p))
        .unwrap_or_else(|_| p.to_path_buf());
    match platform {
        ShellPlatform::Unix => canonical,
        ShellPlatform::Windows => crate::util::strip_verbatim_prefix(&canonical),
    }
}

/// True when a raw shell word contains an expansion (`$`, backtick, `$(`,
/// `$'…'`) that cannot be resolved statically (unquoted or double-quoted;
/// single quotes suppress).
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
// execs the real grep in place (the sentinel is reserved for a run that
// already wrote output and cannot be handed back).
#[serde(default)]
#[expect(non_snake_case, clippy::struct_excessive_bools)] // flag-letter names, grep CLI surface
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

/// Normalized single-string flag surface for a served grep (telemetry),
/// emitted in `GrepFlags` field order with one exception: when both
/// before/after are set they fold into a single `C` (e.g. `-B2 -A3` →
/// "C", not "BA").
fn flags_surface(flags: &GrepFlags) -> String {
    let mut s = String::new();
    if flags.n {
        s.push('n');
    }
    if flags.i {
        s.push('i');
    }
    if flags.v {
        s.push('v');
    }
    if flags.w {
        s.push('w');
    }
    if flags.x {
        s.push('x');
    }
    if flags.a {
        s.push('a');
    }
    if flags.h {
        s.push('h');
    }
    if flags.H {
        s.push('H');
    }
    if flags.s {
        s.push('s');
    }
    if flags.r {
        s.push('r');
    }
    if flags.o {
        s.push('o');
    }
    if flags.null {
        s.push('z');
    }
    if flags.c {
        s.push('c');
    }
    if flags.l {
        s.push('l');
    }
    if flags.m.is_some() {
        s.push('m');
    }
    if flags.before > 0 && flags.after > 0 {
        s.push('C');
    } else {
        if flags.before > 0 {
            s.push('B');
        }
        if flags.after > 0 {
            s.push('A');
        }
    }
    s
}

/// A word from the grep segment: its unquoted value plus redirect metadata.
#[derive(Clone)]
struct GrepWord {
    /// Unquoted value (expansions already rejected).
    value: String,
    /// Raw spelling for redirects preserved verbatim in the rewrite.
    raw: String,
    redirect: bool,
    needs_target: bool,
}

/// True when `token` already opens a redirect spelling `sh` reads as one
/// operator plus target — an optional fd digit prefix included. A later
/// operator byte inside such a token stays in it: the rewrite re-emits the raw
/// spelling verbatim, so the shell re-reads it exactly as it read the original
/// line. The canonical full-token reading is [`super::scan::classify_shell_token`]
/// (which the tokenizer's flush calls); this is the partial-token test the
/// mid-word arms need, and it must keep answering the same syntax — the
/// tokenizer's redirect pins catch the drift.
fn opens_redirect(token: &str) -> bool {
    let rest = token.trim_start_matches(|c: char| c.is_ascii_digit());
    if rest.len() == token.len() {
        // No fd prefix: the token opens with the operator as written (`&>` is
        // one operator of its own).
        token.starts_with(['<', '>']) || token.starts_with("&>")
    } else {
        // An fd prefix: only `<`/`>` may follow it — `2&>log` is the word `2`
        // plus `&>log`, bash's own reading of that spelling.
        rest.starts_with(['<', '>'])
    }
}

/// Tokenize a grep segment (after the verb): quote-aware split, unquote,
/// redirect classification (delegated to the read-only guard's token
/// classifier — single source of truth for redirect-token semantics). The
/// reading is the platform shell's own: `sh`'s word splitting on unix and
/// cmd.exe's (no escapes, `'` ordinary) on Windows.
fn grep_tokenize(segment: &str, platform: ShellPlatform) -> Result<Vec<GrepWord>, Fallback> {
    if platform == ShellPlatform::Windows {
        let words = windows::tokenize(segment)
            .ok_or_else(|| Fallback::CmdSyntax("unbalanced quotes".into()))?;
        // cmd.exe splits a word at an unquoted redirect operator glued to it,
        // while the shared classifier reads only an operator that OPENS the
        // token — so a word like `x>out.txt` would be served as one pattern
        // with the redirect dropped, a search of different text. Such a member
        // is refused instead. An operator glued to the *verb* (`grep>x f.txt`)
        // never reaches this test: `grep>x` is no grep verb, so the line carries
        // no member to refuse and is left to the platform (the module header's
        // exemption list). Windows only: the unix lane splits the glued spelling
        // out of the word itself (the module header's word reading), which cmd's
        // own reader is not modelled for.
        if let Some(word) = words
            .iter()
            .find(|w| !w.redirect && windows::has_glued_redirect(&w.raw))
        {
            return Err(Fallback::CmdSyntax(format!("redirect in `{}`", word.raw)));
        }
        return Ok(words);
    }
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
        let value = unquote_word(&raw, platform)?;
        let (redirect, needs_target) = match super::scan::classify_shell_token(&raw) {
            super::scan::TokenKind::Regular => (false, false),
            super::scan::TokenKind::Redirect { needs_target } => (true, needs_target),
        };
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
            // An unquoted, unescaped `<`/`>` ends the word before it and starts
            // a redirection, exactly as `sh` reads it. The raw spelling then
            // rides the rewrite verbatim, so the shell performs the write.
            if c == '<' || c == '>' {
                // `>(…)`/`<(…)` is process substitution: a word the shell
                // builds from a command, never a redirect this engine could
                // resolve — the member stays on the real `grep`.
                if chars.peek() == Some(&'(') {
                    return Err(Fallback::ProcessSubstitution);
                }
                // POSIX IO_NUMBER: the digits before the operator belong to it
                // only when the whole word so far is digits (`2>log` redirects
                // fd 2; `x2>log` is the word `x2` plus a redirect). An empty
                // word (the operator opens the token) is vacuously all digits:
                // there is nothing to flush either way.
                let all_digits = current.bytes().all(|b| b.is_ascii_digit());
                if !all_digits && !opens_redirect(&current) {
                    flush(&mut current, &mut out)?;
                }
                current.push(c);
                // A glued `&` (`>&`, `<&`) needs no arm of its own: `&` is never
                // a word boundary here, so it rides the ordinary path into the
                // same token — as does the second operator byte of `>&>`, which
                // the `&>` arm below keeps in it through `opens_redirect`.
                continue;
            }
            // `&>` (and `&>>`) is one bash operator; the word before it stays
            // part of the search.
            if c == '&' && chars.peek() == Some(&'>') {
                if !opens_redirect(&current) {
                    flush(&mut current, &mut out)?;
                }
                current.push(c);
                current.push(chars.next().expect("peeked `>`"));
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
///
/// On Windows this is cmd.exe's own unquoting, which expands nothing: `$` and
/// backticks stay literal (a `$` is a common ERE anchor) and the `%…%`
/// reading belongs to the word scan, which fails closed on it separately.
fn unquote_word(raw: &str, platform: ShellPlatform) -> Result<String, Fallback> {
    if platform == ShellPlatform::Windows {
        return Ok(windows::unquote_word(raw));
    }
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

/// Member-side redirects that move stderr off the shell tool's captured fd 2
/// (into the pipeline or a file): the stream-size marker would corrupt the
/// pipeline output or pollute the file, so stdin serves with such redirects
/// skip it. `1>&2`/`>&2` (stdout→stderr) and scratch-fd dups (`3>&1`) keep
/// stderr captured and are not matched; `2>file` is broader than the
/// pipe-corruption risk but the marker is uncapturable once fd 2 leaves the
/// capture (and would pollute the file). Digit-fd dups like `12>&1` contain
/// `2>` and suppress too — fail-closed, telemetry loss only.
fn redirect_merges_stderr(r: &str) -> bool {
    r.contains("2>") || r.contains("&>")
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
    /// Raw operand token spellings (quote/glob checks run on these).
    operand_tokens: Vec<String>,
    /// Redirect tokens preserved verbatim in the rewritten segment.
    redirects: Vec<String>,
    /// `verb` + normalized flags + patterns (operands appended later).
    fallback_prefix: Vec<String>,
}

#[expect(clippy::too_many_lines)] // BSD-getopt flag table is inherently long
/// Parse a grep segment's words (after the verb) with BSD-getopt semantics:
/// GNU-style permutation, value-taking options consuming the rest of their
/// token, last-wins -G/-E/-F, the macOS option cluster, and the verified flag
/// surface only — anything else falls back.
fn parse_grep_words(
    words: &[GrepWord],
    verb: &str,
    platform: ShellPlatform,
) -> Result<ParsedGrep, Fallback> {
    // Redirects (and their targets) are removed by the shell before grep sees
    // the argv; collect them verbatim for the rewrite.
    let mut argv: Vec<GrepWord> = Vec::new();
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
            argv.push(words[i].clone());
        }
        i += 1;
    }
    // cmd.exe expands `%…%` before the program sees its argv — a pattern, an
    // `-e` value and a filter spelling included, not just an operand — so a word
    // carrying a pair cannot be served with its literal text (see
    // [`windows::has_percent_expansion`] for the per-word reading and why it is
    // enough here). A lone `%` and every `!` are ordinary characters here: the
    // interpreter this process spawns is `cmd /C` without `/V:ON`. Redirects are
    // exempt: the rewrite keeps them verbatim, so cmd.exe expands them exactly
    // as it would have in the original command.
    if platform == ShellPlatform::Windows
        && let Some(word) = argv.iter().find(|w| windows::has_percent_expansion(&w.raw))
    {
        return Err(Fallback::Expansion(word.raw.clone()));
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
        if !options_ended && tok.value == "--" {
            options_ended = true;
            i += 1;
            continue;
        }
        if !options_ended && tok.value.starts_with('-') && tok.value.len() > 1 {
            if let Some(rest) = tok.value.strip_prefix("--") {
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
                    let v = argv
                        .get(i)
                        .map(|w| w.value.clone())
                        .ok_or(Fallback::MissingOptionValue)?;
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
                let chars: Vec<char> = tok.value[1..].chars().collect();
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
                                argv.get(i)
                                    .map(|w| w.value.clone())
                                    .ok_or(Fallback::MissingOptionValue)?
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
            positional = Some(tok.value.clone());
            i += 1;
        } else {
            operand_tokens.push(tok.raw.clone());
            i += 1;
        }
    }

    // Flag-surface validation: under -c/-l, -o is inert and
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

/// Escapes whose BSD-grep semantics differ from the engine (word anchors and
/// PCRE-style classes): shared by the ERE pre-check and the BRE translator so
/// the reject list cannot drift between the two paths.
fn bsd_escaped_class_reject(c: char) -> Option<&'static str> {
    match c {
        '<' | '>' => Some(r"\< \>"),
        's' | 'S' | 'w' | 'W' | 'd' | 'D' => Some(r"\s\w\d"),
        _ => None,
    }
}

/// ERE passes through, but reject constructs whose engine semantics differ
/// from BSD (backrefs and unknown escapes are caught by compile validation).
fn check_ere_safe(pattern: &str) -> Result<(), Fallback> {
    let mut escaped = false;
    for c in pattern.chars() {
        if escaped {
            if let Some(what) = bsd_escaped_class_reject(c) {
                return Err(Fallback::Pattern(what.into()));
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
                esc if let Some(what) = bsd_escaped_class_reject(esc) => {
                    return Err(Fallback::Pattern(what.into()));
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
        .unicode(true)
        .line_terminator(Some(b'\n'));
    if mode == MatchMode::Fixed {
        b.fixed_strings(true);
    }
    b.build_many(patterns)
}

/// Matcher wrapper: BSD grep under C.UTF-8 treats any line containing invalid
/// UTF-8 as a silent non-match for every matcher mode (-F/-E/-i/-v alike) —
/// except in binary files, where a NUL in the first 32 KiB switches grep to
/// byte-oriented matching and invalid-UTF-8 match lines still count. The
/// wrapper rejects invalid haystacks when `validate_utf8` is set (text files).
/// With a configured line terminator, the inner matcher engages the whole-buffer
/// SIMD prefilter (grep-regex's `fast_line_regex`/`find_candidate_line`); the
/// `find_candidate_line` override re-validates `Confirmed` lines so invalid-UTF-8
/// lines do not produce false matches (the searcher reports `Confirmed` lines
/// without calling `find_at`).
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
        self.inner.line_terminator()
    }

    fn non_matching_bytes(&self) -> Option<&grep_matcher::ByteSet> {
        self.inner.non_matching_bytes()
    }

    fn find_candidate_line(
        &self,
        haystack: &[u8],
    ) -> Result<Option<grep_matcher::LineMatchKind>, Self::Error> {
        let Some(kind) = self.inner.find_candidate_line(haystack)? else {
            return Ok(None);
        };
        // Binary files are byte-oriented: the inner candidate/confirmed result
        // stands on its own.
        if !self.validate_utf8 {
            return Ok(Some(kind));
        }
        match kind {
            // The searcher verifies Candidate lines via `is_match` → `find_at`,
            // which already rejects invalid-UTF-8 lines.
            grep_matcher::LineMatchKind::Candidate(i) => {
                Ok(Some(grep_matcher::LineMatchKind::Candidate(i)))
            }
            // The searcher reports Confirmed lines without calling `find_at`, so
            // an invalid-UTF-8 line would be a false match. Re-validate the line
            // and, when it is invalid, skip past it and keep searching. Iterative
            // (not recursive): a large non-UTF-8 file littered with literal-
            // matching invalid lines would otherwise build a lengthy call chain
            // and overflow the worker stack.
            grep_matcher::LineMatchKind::Confirmed(i) => {
                let (start, end) = locate_line(haystack, b'\n', i);
                if std::str::from_utf8(&haystack[start..end]).is_ok() {
                    return Ok(Some(grep_matcher::LineMatchKind::Confirmed(i)));
                }
                let mut pos = end;
                loop {
                    let Some(kind) = self.inner.find_candidate_line(&haystack[pos..])? else {
                        return Ok(None);
                    };
                    match kind {
                        grep_matcher::LineMatchKind::Candidate(j) => {
                            return Ok(Some(grep_matcher::LineMatchKind::Candidate(pos + j)));
                        }
                        grep_matcher::LineMatchKind::Confirmed(j) => {
                            let (start, end) = locate_line(haystack, b'\n', pos + j);
                            if std::str::from_utf8(&haystack[start..end]).is_ok() {
                                return Ok(Some(grep_matcher::LineMatchKind::Confirmed(pos + j)));
                            }
                            pos = end;
                        }
                    }
                }
            }
        }
    }
}

/// Locate the line (terminator included) containing the byte offset `pos`,
/// mirroring the (private) `grep_searcher::lines::locate` for a single point.
fn locate_line(bytes: &[u8], term: u8, pos: usize) -> (usize, usize) {
    let start = bytes[..pos]
        .iter()
        .rposition(|&b| b == term)
        .map_or(0, |i| i + 1);
    let end = bytes[pos..]
        .iter()
        .position(|&b| b == term)
        .map_or(bytes.len(), |i| pos + i + 1);
    (start, end)
}

// ── Operand resolution and the serve gate ─────────────────────────────────

/// Pipeline context for a served grep member.
#[derive(Clone, Copy)]
struct PipelineCtx {
    /// The member feeds a pipeline tail: its stdout must not be capped, so
    /// downstream members see the full stream (byte-identity with grep).
    piped: bool,
    /// The member is not the pipeline's first member: it is fed by the
    /// producer's stdout via stdin.
    stdin_fed: bool,
    /// The member's stderr still reaches the shell tool's capture: no
    /// shell-level `exec` stderr merge before it. When false, the stream-size
    /// marker is suppressed (it would leak into the agent-visible stdout).
    marker_ok: bool,
}

/// Serve one grep segment: parse, translate, resolve operands, apply the
/// serve-worthiness gate, and build the spec plus its member-side redirect
/// tokens (rendered into the rewrite by [`join_rewritten`]). A non-first
/// pipeline member (`ctx.stdin_fed`) is served from the producer's stdin when
/// it has no operands and no -r (both stay on the fallback). `allow_single`
/// bypasses the single-file perf gate: `serve_command` sets it on Windows, where
/// there is no host grep to keep a single-file lookup on.
fn serve_one_grep(
    segment: &str,
    verb: &str,
    cwd: &Path,
    home: &Path,
    platform: ShellPlatform,
    allow_single: bool,
    ctx: PipelineCtx,
) -> Result<(EngineSpec, Vec<String>), Fallback> {
    let mut words = grep_tokenize(segment, platform)?;
    // The segment starts with the verb; the parser sees only its arguments.
    // The comparison goes back through the family normalizer, since the verb
    // arrived normalized (on Windows `GREP.EXE` is served as `grep`).
    if words
        .first()
        .is_some_and(|w| grep_family(&w.value, platform) == Some(verb))
    {
        words.remove(0);
    } else {
        return Err(Fallback::NestedGrep);
    }
    let parsed = parse_grep_words(&words, verb, platform)?;

    let mut operands: Vec<Operand> = Vec::new();
    if parsed.operand_tokens.is_empty() {
        if parsed.flags.r {
            if ctx.stdin_fed {
                // `grep -r pat` with no operands walks the cwd and ignores
                // stdin (BSD) — never stdin-feed it.
                return Err(Fallback::StdinRecursive);
            }
            // `grep -r pat` with no operands walks the cwd (BSD behavior).
            operands.push(Operand {
                display: ".".into(),
                resolved: cwd.to_string_lossy().into_owned(),
                trailing_slash: false,
            });
        } else if ctx.stdin_fed {
            // Producer-fed stdin serve (no operands).
        } else {
            return Err(Fallback::StdinMode);
        }
    } else {
        if ctx.stdin_fed {
            // Non-first grep with file operands (incl. "-") is a deliberate
            // scope cut — reject (BSD ignores stdin when operands exist).
            return Err(Fallback::StdinOperands);
        }
        for tok in &parsed.operand_tokens {
            if unquote_word(tok, platform)? == "-" {
                return Err(Fallback::StdinMode);
            }
            let expanded = resolve_operand(tok, cwd, home, platform)?;
            if expanded.is_empty() {
                return Err(Fallback::UnexpandableGlob);
            }
            operands.extend(expanded);
        }
    }

    // Serve gate: recursive walks (the expensive case), or multi-file
    // invocations. A single-file lookup stays on the real grep on unix — the
    // host binary is faster — and `allow_single` (Windows, where there is no
    // host grep) is what serves it there instead.
    // stdin-fed serves bypass the gate (the stream size is unknowable and
    // the ~6 ms engine tax is invisible against the producer's run).
    let mut dir_count = 0usize;
    for op in &operands {
        if fs::metadata(&op.resolved).is_ok_and(|m| m.is_dir()) {
            dir_count += 1;
        }
    }
    let serve = ctx.stdin_fed
        || allow_single
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
        piped: ctx.piped,
        stdin: ctx.stdin_fed,
        // The stream-size marker would leak into the tool's captured stdout
        // (a shell-level `exec 2>&1` before the member) or corrupt/pollute a
        // member-side stderr redirect — skip it then.
        report_stream_bytes: ctx.stdin_fed
            && ctx.marker_ok
            && !parsed.redirects.iter().any(|r| redirect_merges_stderr(r)),
    };

    Ok((spec, parsed.redirects))
}

/// Resolve one raw operand token into concrete operands (tilde, glob
/// expansion, relative-to-cwd). Quote/glob checks run on the token exactly
/// as typed: quoted/escaped `~` and glob metacharacters are literal
/// filenames to BSD grep and must not expand. Unresolvable forms fall back.
///
/// The `~` arms are unix-only: cmd.exe expands no `~`, so on Windows it is an
/// ordinary directory name and resolves relative to the cwd. The `%`/`!`
/// reading is checked once, for every word of the member, in
/// [`parse_grep_words`].
fn resolve_operand(
    tok: &str,
    cwd: &Path,
    home: &Path,
    platform: ShellPlatform,
) -> Result<Vec<Operand>, Fallback> {
    if platform == ShellPlatform::Unix {
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
    }
    if has_unquoted_glob(tok, platform) {
        let value = unquote_word(tok, platform)?;
        // `~/` expands to home only when the raw token opens with an unquoted
        // `~/`; `"~"/*.txt` (quoted tilde + unquoted glob) is a literal
        // cwd-relative path, and the unquoted value would wrongly home-strip.
        let pattern = if platform == ShellPlatform::Unix && tok.starts_with("~/") {
            home.join(&value[2..]).to_string_lossy().into_owned()
        } else {
            value
        };
        let matches = expand_glob(&pattern, cwd, platform)?;
        if matches.is_empty() {
            return Err(Fallback::UnexpandableGlob);
        }
        return Ok(matches
            .iter()
            .map(|m| {
                let abs = absolute_or_cwd(cwd, m, platform);
                operand_from_path(m, &abs, false)
            })
            .collect());
    }
    if platform == ShellPlatform::Unix
        && let Some(rest) = tok.strip_prefix("~/")
    {
        let expanded = home.join(unquote_word(rest, platform)?);
        return Ok(vec![operand_from_path(
            &expanded.to_string_lossy(),
            &expanded,
            false,
        )]);
    }
    let value = unquote_word(tok, platform)?;
    let trailing_slash = match platform {
        ShellPlatform::Unix => value.ends_with('/'),
        ShellPlatform::Windows => value.ends_with(['\\', '/']),
    };
    let abs = absolute_or_cwd(cwd, &value, platform);
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
/// Quote- and escape-aware via [`super::track_char_context`] — bash semantics:
/// glob chars inside single OR double quotes are literal filenames. Under
/// cmd.exe only `"` quotes (and there is no escape character at all), so the
/// Windows reading is the `windows` layer's.
fn has_unquoted_glob(tok: &str, platform: ShellPlatform) -> bool {
    if platform == ShellPlatform::Windows {
        return windows::has_unquoted_glob(tok);
    }
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    for c in tok.chars() {
        if super::track_char_context(c, &mut in_single, &mut in_double, &mut escaped)
            && matches!(c, '*' | '?' | '[')
        {
            return true;
        }
    }
    false
}

/// Shell-style glob expansion for operand tokens: dotfile exclusion, bracket
/// expressions, no-match → empty (the caller falls back). The returned
/// display strings match what the shell would pass to grep. Home expansion of
/// `~/…` is applied by the caller (only for an unquoted leading `~/`).
///
/// The walk root is the platform's own: `/` (or the cwd) and `/`-separated
/// components on unix; a drive, UNC share or rooted separator (or the cwd) and
/// `\`-separated components on Windows.
fn expand_glob(
    pattern: &str,
    cwd: &Path,
    platform: ShellPlatform,
) -> Result<Vec<String>, Fallback> {
    if platform == ShellPlatform::Windows {
        if pattern.ends_with(['\\', '/']) {
            return Err(Fallback::UnexpandableGlob);
        }
        let (prefix, comps) =
            windows::split_components(pattern).ok_or(Fallback::UnexpandableGlob)?;
        let base = if prefix.is_empty() {
            cwd.to_path_buf()
        } else {
            PathBuf::from(&prefix)
        };
        let comps: Vec<&str> = comps.iter().map(String::as_str).collect();
        let mut results = Vec::new();
        glob_walk(&base, &prefix, &comps, &mut results, platform);
        results.sort();
        return Ok(results);
    }
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
    glob_walk(&base, display_prefix, &comps, &mut results, platform);
    // The shell sorts glob expansions (LC_ALL=C.UTF-8 → byte order); grep
    // emits operands in command-line order, so operand order must match.
    results.sort();
    Ok(results)
}

fn glob_walk(
    dir: &Path,
    display: &str,
    comps: &[&str],
    results: &mut Vec<String>,
    platform: ShellPlatform,
) {
    let Some((first, rest)) = comps.split_first() else {
        return;
    };
    if first.contains(['*', '?', '[']) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            // Shell glob semantics per component: fnmatch with FNM_PERIOD
            // (leading dot must match a literal dot).
            if !fnmatch_flags(first, &name, true, platform) {
                continue;
            }
            let child_display = join_display(display, &name, platform);
            if rest.is_empty() {
                results.push(child_display);
            } else if entry.path().is_dir() {
                glob_walk(&entry.path(), &child_display, rest, results, platform);
            }
        }
    } else {
        let next = dir.join(first);
        let child_display = join_display(display, first, platform);
        if rest.is_empty() {
            if next.exists() {
                results.push(child_display);
            }
        } else if next.is_dir() {
            glob_walk(&next, &child_display, rest, results, platform);
        }
    }
}

/// Join one glob component onto the display prefix it was matched under: with
/// `/` on unix (where a lone `/` prefix is the root) and with `\` on Windows
/// (where the prefix of [`windows::split_components`] already carries its own
/// separator).
fn join_display(prefix: &str, name: &str, platform: ShellPlatform) -> String {
    match platform {
        ShellPlatform::Windows => windows::join_display(prefix, name),
        ShellPlatform::Unix => {
            if prefix.is_empty() {
                name.to_string()
            } else if prefix == "/" {
                format!("/{name}")
            } else {
                format!("{prefix}/{name}")
            }
        }
    }
}

// ── Engine side: hidden subcommand execution ──────────────────────────────

/// Entry point for the hidden `__grep-engine` subcommand (dispatched from
/// `main()` before instance-lock acquisition). `--probe` answers whether the
/// binary still carries the subcommand (self-update version skew). Any runtime
/// doubt hands the member back to the real grep — `exec`ing it in place on
/// unix, and on Windows (where there is no system `grep` to replace ourselves
/// with) reporting the reason and exiting with the sentinel code the parent
/// refuses the call on.
pub fn run_engine(args: &[String]) -> i32 {
    if args.first().map(String::as_str) == Some("--probe") {
        return 0;
    }
    // Rust ignores SIGPIPE by default; grep dies on a closed pipe (exit 141).
    // With the default disposition, `grep | head` would scan the whole input.
    // Windows has no SIGPIPE — there the same closed pipe surfaces as the write
    // error [`exit_on_broken_pipe`] turns into the same exit.
    // SAFETY: restoring the default SIGPIPE disposition is process-global and
    // idempotent; the engine process exists only to serve this one invocation.
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    let json = match read_spec(args) {
        Ok(json) => json,
        Err(message) => return engine_failed(Some(&message)),
    };
    let spec: EngineSpec = match serde_json::from_str(&json) {
        Ok(s) => s,
        Err(e) => return engine_failed(Some(&format!("grep: engine: bad spec: {e}"))),
    };
    if spec.version != PROTOCOL_VERSION {
        // Self-update swapped the binary mid-run: nothing is written yet, so
        // the real grep is the correct answer.
        return cannot_serve(&spec, "spec version mismatch");
    }
    // `serve` itself reports panics: nothing-written-or-read panics return Err
    // (the real grep is still available); post-output panics return the
    // sentinel. This outer catch_unwind is the last-resort net for panics
    // outside `serve`'s scope.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| serve(&spec)));
    match result {
        Ok(Ok(code)) if code != ENGINE_FAILED_EXIT => code,
        // Err only from pre-output checks (cwd mismatch, matcher build, or a
        // panic before anything was written or read) — the real grep is clean.
        Ok(Err(reason)) => cannot_serve(&spec, reason),
        // `serve` panicked with output already streamed (it reports the sentinel
        // then), or a panic escaped it entirely: partial output may exist, so
        // refusing is the only honest answer.
        Ok(Ok(_)) | Err(_) => engine_failed(None),
    }
}

/// Report an engine failure the parent must treat as an unserved search: the
/// one-line `detail` (when there is one) and, on the platform whose parent
/// reads it, [`ENGINE_REFUSAL_MARKER`] on a line of its own — so a refusal
/// survives a pipeline tail masking the sentinel exit status.
fn engine_failed(detail: Option<&str>) -> i32 {
    if SHELL_PLATFORM == ShellPlatform::Windows {
        // First, so a stream that hits the parent's stderr cap still carries it.
        eprintln!("{ENGINE_REFUSAL_MARKER}");
    }
    if let Some(detail) = detail {
        eprintln!("{detail}");
    }
    ENGINE_FAILED_EXIT
}

/// Read the spec JSON the member was invoked with: a bare JSON argument on
/// unix, or the scratch file named by [`windows::SPEC_FILE_FLAG`] on Windows,
/// where the command line cannot carry the payload (see the module header).
/// The `Err` is the one-line reason printed before the sentinel exit.
fn read_spec(args: &[String]) -> Result<String, String> {
    match args {
        [flag, path, ..] if flag == windows::SPEC_FILE_FLAG => {
            fs::read_to_string(path).map_err(|e| format!("grep: engine: spec file {path}: {e}"))
        }
        [json, ..] => Ok(json.clone()),
        [] => Err("grep: engine: missing spec".to_string()),
    }
}

/// The real grep is about to produce the answer instead of the engine, because
/// the engine cannot serve this spec (cwd divergence, matcher build failure,
/// protocol mismatch, a panic before any output).
///
/// Unix: replace the process with the real grep — re-exec is sound only while
/// stdin was not yet consumed (the cwd/version/matcher pre-checks run before
/// any stdin read), and the PID/PGID are preserved, so the shell tool's
/// process-group timeout kill still applies.
///
/// Windows: cmd.exe resolves `grep` through `PATHEXT` and its own cwd, and the
/// engine has no way to re-exec the exact program the member would have run,
/// so the reason goes to stderr and the sentinel exit refuses the call — the
/// parent reports that line to the agent as the cause.
///
/// The `cfg` below is not a platform policy branch (that is
/// [`SHELL_PLATFORM`]): it guards `exec`, an API the target simply does not
/// have off unix.
fn cannot_serve(spec: &EngineSpec, reason: &str) -> i32 {
    #[cfg(unix)]
    {
        let _ = reason;
        exec_grep(&spec.fallback)
    }
    #[cfg(not(unix))]
    {
        engine_failed(Some(&format!("{}: engine: {reason}", spec.verb)))
    }
}

/// Replace the engine process with the real grep: PID/PGID preserved, so the
/// shell tool's process-group timeout kill still applies. Unix-only — the
/// Windows hand-back is the sentinel exit (see [`cannot_serve`]).
#[cfg(unix)]
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
/// per-file grep semantics (or read stdin for stdin-fed members). Returns the
/// aggregate exit code (0/1/2).
///
/// `Err(reason)` means nothing was written and stdin was not consumed, so the
/// caller can hand the member back to the real grep (cwd divergence, matcher
/// build failure, or a panic before any input/output). A panic after output
/// started or after the stdin head read returns `Ok(ENGINE_FAILED_EXIT)` —
/// running grep then would append a false result or read the drained pipe
/// remainder, so the parent re-runs.
fn serve(spec: &EngineSpec) -> Result<i32, &'static str> {
    let actual_cwd = std::env::current_dir().map_err(|_| "working directory unavailable")?;
    // The parent's tracked cwd and this process's must name the same directory.
    // Unix compares the two canonical spellings byte-exactly; Windows resolves
    // this process's own cwd through the same canonicalization the parent used —
    // the platform's path identity (case, separators, the verbatim prefix) plus
    // the aliases `fs::canonicalize` follows (a junction or a `subst` drive would
    // otherwise fail the gate for every member).
    let same_cwd = match SHELL_PLATFORM {
        ShellPlatform::Unix => actual_cwd == Path::new(&spec.cwd),
        ShellPlatform::Windows => windows::same_directory(
            &canonical_or_lexical(&actual_cwd, ShellPlatform::Windows),
            Path::new(&spec.cwd),
        ),
    };
    if !same_cwd {
        // The cd chain diverged at runtime (e.g. a failing `cd` in a chain) —
        // the real grep in the actual cwd is the authentic result.
        return Err("working directory diverged from the analyzed command");
    }
    let matcher = build_matcher(&spec.patterns, spec.mode, &spec.flags)
        .map_err(|_| "pattern could not be compiled")?;
    let mut out = Output::new(
        OutputSink::Stdout(io::BufWriter::with_capacity(16 * 1024, io::stdout())),
        output_limit(spec),
    );
    let stdin_consumed = std::cell::Cell::new(false);
    let stdin = io::stdin();
    let code = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        serve_into(spec, &matcher, &mut out, stdin.lock(), &stdin_consumed)
    }));
    match code {
        Ok(code) => {
            out.finish();
            Ok(code)
        }
        // A panic before anything was written or read: exec'ing the real grep
        // is clean, and its exit code cannot be masked by a pipe member.
        Err(_) if out.written == 0 && !stdin_consumed.get() => {
            Err("engine panicked before producing output")
        }
        // A panic mid-search may have streamed partial output or consumed the
        // producer's pipe; exec'ing grep would append a false result. Flush
        // and return the sentinel: the parent discards this run (unix re-runs
        // the original command, Windows refuses the call).
        Err(_) => {
            out.finish();
            Ok(ENGINE_FAILED_EXIT)
        }
    }
}

/// Serve a spec writing into caller-owned buffers (testable in-process). The
/// stdin reader is injectable so in-process tests never touch the process's
/// real stdin.
fn serve_into<R: io::Read>(
    spec: &EngineSpec,
    matcher: &grep_regex::RegexMatcher,
    out: &mut Output,
    stdin: R,
    stdin_consumed: &std::cell::Cell<bool>,
) -> i32 {
    if spec.stdin {
        return serve_stdin(spec, matcher, out, stdin, stdin_consumed);
    }
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

/// Byte-counting reader wrapper for stream-size telemetry.
struct CountingReader<R> {
    inner: R,
    count: u64,
}

impl<R: io::Read> CountingReader<R> {
    fn new(inner: R) -> Self {
        CountingReader { inner, count: 0 }
    }

    fn count(&self) -> u64 {
        self.count
    }
}

impl<R: io::Read> io::Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.count += n as u64;
        Ok(n)
    }
}

/// Read the 32 KiB binary-window head (BSD's binary sniffing window), returned
/// truncated to what was read, with the NUL verdict.
fn read_binary_head<R: io::Read>(src: &mut R) -> io::Result<(Vec<u8>, bool)> {
    let mut head = vec![0u8; BINARY_WINDOW];
    let n = src.read(&mut head)?;
    head.truncate(n);
    let binary = head.contains(&0);
    Ok((head, binary))
}

/// Searcher with the spec's matching surface. Context flags are suppressed in
/// count modes; -m caps matches (stopping input consumption on streams).
fn build_searcher(spec: &EngineSpec) -> (grep_searcher::Searcher, bool) {
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
    (sb.build(), count_mode)
}

/// Search a (possibly non-seekable) stream with the spec's BSD sink; `binary`
/// is the NUL-window verdict. Returns the search result and whether any line
/// matched (count modes finish their count inside). The sink borrows `out`;
/// the caller writes the stream-size marker after this returns — its stderr
/// position relative to grep errors is irrelevant (the parent's strip scans
/// the whole buffer).
fn search_stream<R: io::Read>(
    spec: &EngineSpec,
    matcher: &grep_regex::RegexMatcher,
    display: &str,
    show_prefix: bool,
    binary: bool,
    input: &mut R,
    out: &mut Output,
) -> (Result<(), io::Error>, bool) {
    let search_matcher = SearchMatcher {
        inner: matcher.clone(),
        validate_utf8: !binary,
    };
    let (mut searcher, count_mode) = build_searcher(spec);
    let matcher_for_search = search_matcher.clone();
    let (result, selected_any) = {
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
        let result = searcher.search_reader(matcher_for_search, input, &mut sink);
        if count_mode {
            sink.finish_count();
        }
        (result, sink.selected_any)
    };
    (result, selected_any)
}

/// Serve a stdin-fed grep member: read the producer's pipe with BSD binary
/// semantics — a 32 KiB NUL-sniff buffer, filled by one partial read (a pipe
/// yields only its available bytes, and BSD sniffs the same way), the
/// "(standard input)" display name, --include/--exclude/--exclude-dir ignored
/// on stdin, per-line flush, and -m stopping the read (grep-searcher's
/// max_matches, matching BSD's instant exit).
fn serve_stdin<R: io::Read>(
    spec: &EngineSpec,
    matcher: &grep_regex::RegexMatcher,
    out: &mut Output,
    mut stdin: R,
    stdin_consumed: &std::cell::Cell<bool>,
) -> i32 {
    let (head, binary) = match read_binary_head(&mut stdin) {
        Ok(v) => v,
        Err(e) => {
            emit_error(spec, out, "(standard input)", &e.to_string());
            return 2;
        }
    };
    stdin_consumed.set(true);
    // Non-seekable stream: buffer the head, then read the remainder (the same
    // pattern that serves FIFOs and /dev/* — BSD grep likewise never seeks).
    let mut input = CountingReader::new(io::Cursor::new(&head).chain(stdin));
    let (search, selected_any) = search_stream(
        spec,
        matcher,
        "(standard input)",
        spec.flags.H && !spec.flags.h,
        binary,
        &mut input,
        out,
    );
    if spec.report_stream_bytes {
        out.write_err(&format!(
            "{}: {STREAM_SIZE_MARKER}: {}\n",
            spec.verb,
            input.count()
        ));
    }
    if let Err(e) = search {
        emit_error(spec, out, "(standard input)", &e.to_string());
        return 2;
    }
    i32::from(!selected_any)
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
                    if dir_excluded(display, &spec.exclude_dir, SHELL_PLATFORM) {
                        return OperandResult::NoMatch;
                    }
                    return walk_dir(op, spec, matcher, out, show_prefix);
                }
                if op.trailing_slash {
                    emit_error(spec, out, display, "Not a directory");
                    return OperandResult::Error;
                }
                if !file_allowed_by_filters(display, spec, SHELL_PLATFORM) {
                    return OperandResult::NoMatch;
                }
                return search_file(path, display, spec, matcher, out, show_prefix);
            }
        }
    }

    if lmeta.is_dir() {
        if dir_excluded(display, &spec.exclude_dir, SHELL_PLATFORM) {
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
    if !file_allowed_by_filters(display, spec, SHELL_PLATFORM) {
        return OperandResult::NoMatch;
    }
    search_file(path, display, spec, matcher, out, show_prefix)
}

/// BSD fnmatch semantics for --include/--exclude: patterns match the basename
/// OR the full traversal path, `*` crosses `/`, anchored full-string, and the
/// LAST matching pattern in command-line order decides (include vs exclude).
fn file_allowed_by_filters(display: &str, spec: &EngineSpec, platform: ShellPlatform) -> bool {
    // One ordered list of include+exclude patterns; last match wins. Each
    // pattern matches the basename or the full traversal path.
    let base = display_basename(display, platform);
    let mut last: Option<bool> = None;
    for (include, pat) in &spec.filters {
        if fnmatch(pat, display, platform) || fnmatch(pat, base, platform) {
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
fn dir_excluded(display: &str, exclude_dir: &[String], platform: ShellPlatform) -> bool {
    let base = display_basename(display, platform);
    exclude_dir
        .iter()
        .any(|pat| fnmatch(pat, base, platform) || fnmatch(pat, display, platform))
}

/// The basename of a displayed path, in the platform's own separator reading:
/// `\` separates on Windows and is an ordinary filename character on unix.
fn display_basename(display: &str, platform: ShellPlatform) -> &str {
    match platform {
        ShellPlatform::Unix => display.rsplit('/').next().unwrap_or(display),
        ShellPlatform::Windows => display.rsplit(['/', '\\']).next().unwrap_or(display),
    }
}

/// Shell glob matching, with `period` standing for `FNM_PERIOD` (a leading dot
/// in the name must be matched by a literal dot): the host `fnmatch` on unix
/// — bash globbing passes `FNM_PERIOD`, BSD grep's --include/--exclude calls
/// pass no flags — and the pure-Rust [`windows::fnmatch`] on Windows, which has
/// no libc to call. The two are pinned differentially against each other by
/// the `windows_fnmatch_parity` tests, which run on a unix host — only there
/// is the second implementation available.
fn fnmatch_flags(pattern: &str, s: &str, period: bool, platform: ShellPlatform) -> bool {
    if platform == ShellPlatform::Windows {
        return windows::fnmatch(pattern, s, period);
    }
    #[cfg(unix)]
    {
        let Ok(p) = std::ffi::CString::new(pattern) else {
            return false;
        };
        let Ok(n) = std::ffi::CString::new(s) else {
            return false;
        };
        let flags = if period { libc::FNM_PERIOD } else { 0 };
        // SAFETY: both CStrings are NUL-free and outlive the call.
        unsafe { libc::fnmatch(p.as_ptr(), n.as_ptr(), flags) == 0 }
    }
    #[cfg(not(unix))]
    {
        // No host matcher exists off unix; the unix platform is a test-lane
        // value there, so nothing matches (fail-closed).
        false
    }
}

/// fnmatch without FNM_PATHNAME — exactly what BSD grep calls:
/// `*`/`?` cross `/`, `[...]` classes (incl. POSIX names), anchored full-string.
fn fnmatch(pattern: &str, s: &str, platform: ShellPlatform) -> bool {
    fnmatch_flags(pattern, s, false, platform)
}

/// Recursive walk with rg-default exclusions (hidden, gitignore, .ignore, git
/// exclude, global gitignore) plus BSD include/exclude filters and symlink
/// skipping. Explicit operands never pass through here.
///
/// The SEARCH runs on multiple worker threads: each file is searched into a
/// per-file buffer (no global lock during search, so the search itself is
/// parallel), then the buffered stdout/stderr are merged into the real `out`
/// under a brief mutex and the aggregate `OperandResult` is maxed. In-file
/// order is preserved; cross-file output order is non-deterministic (the
/// accepted worker-scheduling delta). The hidden/gitignore exclusion and the
/// symlink skip are preserved unchanged by the builder configuration. "Hidden"
/// is the platform's own reading: a leading dot everywhere, plus the hidden
/// file attribute on Windows (the `ignore` crate's rule), so a walk there can
/// skip a file a unix one would search.
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
    let limit = output_limit(spec);

    let mut builder = ignore::WalkBuilder::new(&root_abs);
    builder.filter_entry({
        // The filter closure is `'static` (it must OWN its captured state), so
        // it gets its own copy of the root label; the walker's worker closure
        // keeps the original.
        let root_display = root_display.clone();
        move |entry| {
            if entry.file_type().is_some_and(|t| t.is_dir()) && entry.depth() > 0 {
                let display = traversal_display(
                    &root_display,
                    &root_for_filter,
                    entry.path(),
                    SHELL_PLATFORM,
                );
                !dir_excluded(&display, &exclude_dir, SHELL_PLATFORM)
            } else {
                true
            }
        }
    });

    // Shared output + aggregate result; search runs per-file into a buffer, the
    // merge lock is held only for the buffer copy (streams per file, and
    // cross-file order is non-deterministic — the accepted parallel-walk delta).
    // The worker closures capture these OWNED state pieces by reference so the
    // per-thread builder closure can be invoked more than once.
    //
    // Accepted parallel-walk trade-offs (documented, not errors): (1) when a
    // recursive walk's total output exceeds OUTPUT_CAP, which files' lines
    // survive the cap is non-deterministic across workers — only the count is
    // preserved, not operand-order; (2) a piped member (limit None) buffers a
    // single file's matches in the worker buffer before releasing them, so per
    // worker memory is bounded by ONE matching file's output (not a constant),
    // where the serial path streamed per-line. Both are rare (the dominant
    // target/ timeout is a non-piped walk, and `| head` ends the walk early via
    // SIGPIPE once tail-prefixed output is flushed).
    let shared_out = std::sync::Mutex::new(&mut *out);
    let shared_result = std::sync::Mutex::new(OperandResult::NoMatch);
    let root_abs = &root_abs;
    let root_display = &root_display;
    let out_ref = &shared_out;
    let result_ref = &shared_result;

    builder.build_parallel().run(|| {
        let mut local = Output::new(OutputSink::Buffer(Vec::new()), limit);
        Box::new(move |entry| {
            let (buf, err, r) = match entry {
                Ok(e) => {
                    let ft = e.file_type();
                    if ft.is_some_and(|t| t.is_symlink() || t.is_dir()) {
                        return ignore::WalkState::Continue; // grep -r skips symlinks; dirs traversed by the walk
                    }
                    let display =
                        traversal_display(root_display, root_abs, e.path(), SHELL_PLATFORM);
                    if !file_allowed_by_filters(&display, spec, SHELL_PLATFORM) {
                        return ignore::WalkState::Continue;
                    }
                    let r = search_file(e.path(), &display, spec, matcher, &mut local, show_prefix);
                    let (buf, err) = local.take_stdio();
                    (buf, err, r)
                }
                Err(e) => {
                    // Unreadable dir/file during traversal → grep-style error.
                    let (path, message) = walk_error_info(&e);
                    let display = path.as_deref().map_or_else(
                        || op.display.clone(),
                        |p| traversal_display(&op.display, root_abs, p, SHELL_PLATFORM),
                    );
                    emit_error(spec, &mut local, &display, &message);
                    let (buf, err) = local.take_stdio();
                    (buf, err, OperandResult::Error)
                }
            };
            let mut guard = out_ref.lock().unwrap_poison();
            guard.write_bytes(&buf);
            for chunk in err.chunks(4096) {
                guard.write_err(&String::from_utf8_lossy(chunk));
            }
            // Flush so a piped tail (`| head`) sees data as each file completes.
            guard.flush();
            drop(guard);
            let mut gr = result_ref.lock().unwrap_poison();
            *gr = (*gr).max(r);
            ignore::WalkState::Continue
        })
    });

    shared_result.into_inner().unwrap_poison()
}

/// Display path for a walked entry: the operand spelling + relative suffix,
/// joined in the platform's separator so the displayed path is spelled the way
/// the operand was.
fn traversal_display(
    root_display: &str,
    root_abs: &Path,
    entry: &Path,
    platform: ShellPlatform,
) -> String {
    let rel = entry.strip_prefix(root_abs).unwrap_or(entry);
    let rel = rel.to_string_lossy();
    if rel.is_empty() {
        return root_display.to_string();
    }
    match platform {
        // `/`-joined, exactly as this has always rendered on unix.
        ShellPlatform::Unix => format!("{}/{}", root_display.trim_end_matches('/'), rel),
        ShellPlatform::Windows => windows::traversal_display(root_display, &rel),
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
    let (head, binary) = match read_binary_head(&mut file) {
        Ok(v) => v,
        Err(e) => {
            emit_error(spec, out, display, &e.to_string());
            return OperandResult::Error;
        }
    };
    // Non-seekable operands (FIFOs, /dev/*) cannot be re-read; stream the
    // buffered head first, then the remainder — BSD grep likewise never seeks
    // (it reads the window and continues from the stream).
    let mut input = io::Cursor::new(&head).chain(file);
    let (search, selected_any) =
        search_stream(spec, matcher, display, show_prefix, binary, &mut input, out);
    if let Err(e) = search {
        emit_error(spec, out, display, &e.to_string());
        return OperandResult::Error;
    }
    if selected_any {
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

#[expect(clippy::struct_excessive_bools)] // grep CLI flag surface
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
        let content = trim_line_terminator(mat.bytes(), SHELL_PLATFORM);
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
        let content = trim_line_terminator(ctx.bytes(), SHELL_PLATFORM);
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

/// Drop a line's terminator. On Windows a CRLF line's `\r` goes with its `\n`,
/// so a served grep never leaks carriage returns into agent-visible lines (the
/// engine's own terminator stays `\n`, matching the real tool's output); a lone
/// trailing `\r` is content, exactly as on unix — the pair is stripped as a
/// pair.
fn trim_line_terminator(b: &[u8], platform: ShellPlatform) -> &[u8] {
    match platform {
        ShellPlatform::Unix => b.strip_suffix(b"\n").unwrap_or(b),
        ShellPlatform::Windows => b
            .strip_suffix(b"\r\n")
            .or_else(|| b.strip_suffix(b"\n"))
            .unwrap_or(b),
    }
}

// ── Output: bounded, self-capping writer ──────────────────────────────────

/// Streams matched lines to stdout as they are produced (so an early-exit
/// tail stops the search the way it does for the real grep). Non-piped members
/// self-cap at [`OUTPUT_CAP`] bytes after which writing stops but the search
/// continues (exit codes stay correct for huge outputs); piped members are
/// unbounded — the tail consumes the full stream, so truncation would be an
/// invisible wrong answer. A write to a closed pipe kills the process: on unix
/// SIGPIPE at its default disposition does it (exactly like grep), on Windows —
/// which has no SIGPIPE — the `BrokenPipe` write error does (see
/// [`exit_on_broken_pipe`]). Stderr is buffered (small, rare) and flushed on
/// finish.
struct Output {
    sink: OutputSink,
    err: Vec<u8>,
    written: usize,
    /// Cap on written stdout bytes; `None` = unbounded (piped member).
    limit: Option<usize>,
}

enum OutputSink {
    Stdout(io::BufWriter<io::Stdout>),
    // In-memory sink: the parallel-walk per-worker buffers (production) and the
    // macOS-gated parity tests.
    Buffer(Vec<u8>),
}

/// The closed-pipe exit shared by both stdout writers: on unix SIGPIPE at its
/// default disposition already ended the process (exactly like grep), so the
/// write error is dropped; on Windows — which has no SIGPIPE — a `BrokenPipe`
/// error is what a closed pipe surfaces as, and it kills the process with
/// [`BROKEN_PIPE_EXIT`].
fn exit_on_broken_pipe(result: &io::Result<()>) {
    #[cfg(not(windows))]
    let _ = result;
    #[cfg(windows)]
    if let Err(err) = result
        && err.kind() == io::ErrorKind::BrokenPipe
    {
        std::process::exit(BROKEN_PIPE_EXIT);
    }
}

impl Output {
    /// New sink with an optional stdout cap (`None` = unbounded piped member).
    fn new(sink: OutputSink, limit: Option<usize>) -> Self {
        Output {
            sink,
            err: Vec::new(),
            written: 0,
            limit,
        }
    }

    fn write_bytes(&mut self, b: &[u8]) {
        let take = match self.limit {
            Some(limit) if self.written < limit => (limit - self.written).min(b.len()),
            Some(_) => return,
            None => b.len(),
        };
        match &mut self.sink {
            OutputSink::Stdout(w) => exit_on_broken_pipe(&w.write_all(&b[..take])),
            OutputSink::Buffer(v) => v.extend_from_slice(&b[..take]),
        }
        self.written += take;
    }

    fn write_byte(&mut self, b: u8) {
        self.write_bytes(&[b]);
    }

    /// Drain the buffered stdout and stderr, leaving the sink empty. For the
    /// buffer sink the stdout Vec is taken; for the stdout sink nothing is
    /// buffered in-process, so the stdout half is empty. Resets `written`
    /// (the per-file cap accounting of the parallel-walk worker buffers).
    fn take_stdio(&mut self) -> (Vec<u8>, Vec<u8>) {
        self.written = 0;
        match &mut self.sink {
            OutputSink::Buffer(v) => (std::mem::take(v), std::mem::take(&mut self.err)),
            OutputSink::Stdout(_) => (Vec::new(), std::mem::take(&mut self.err)),
        }
    }

    /// Error message (stderr) — bounded.
    fn write_err(&mut self, s: &str) {
        if self.err.len() < 64 * 1024 {
            self.err.extend_from_slice(s.as_bytes());
        }
    }

    /// Push buffered stdout to the pipe (per line) so `head` sees data and its
    /// early exit reaches us, through [`exit_on_broken_pipe`].
    fn flush(&mut self) {
        if let OutputSink::Stdout(w) = &mut self.sink {
            exit_on_broken_pipe(&w.flush());
        }
    }

    /// Flush stdout and emit the buffered stderr. The stdout flush goes through
    /// the same closed-pipe rule as every other write (the search is over by
    /// now, so a break here cannot extend the walk, but the policy stays one).
    fn finish(self) {
        if let OutputSink::Stdout(mut w) = self.sink {
            exit_on_broken_pipe(&w.flush());
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
    use super::test_support::{engine_run, engine_run_with_stdin};
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
        // Literal filename operands for quoted/escaped glob and ~ forms:
        // BSD grep searches these literally when the shell quotes/escapes
        // the metacharacters.
        fs::write(ws.join("~"), "needle\n").unwrap();
        fs::write(ws.join("*.txt"), "needle\n").unwrap();
        fs::write(ws.join("a\"b.txt"), "needle\n").unwrap();
        // Quoted `~` + unquoted glob: a literal `~` dir relative to cwd.
        fs::create_dir_all(ws.join("sub/~")).unwrap();
        fs::write(ws.join("sub/~/q.txt"), "needle\n").unwrap();
    }

    /// Fresh temp root with the fixture `ws` and `home` trees built in. Bind
    /// the returned TempDir for the whole test — dropping it deletes the trees.
    fn fixture() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let ws = tmp.path().join("ws");
        let home = tmp.path().join("home");
        fs::create_dir_all(&ws).expect("ws");
        fs::create_dir_all(&home).expect("home");
        build_fixture(&ws, &home);
        (tmp, ws, home)
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
        real_run_fallback_with_stdin(spec, &[])
    }

    /// Run the real grep with explicit stdin bytes (stdin-fed parity rows).
    fn real_run_fallback_with_stdin(spec: &EngineSpec, stdin: &[u8]) -> (Vec<u8>, Vec<u8>, i32) {
        let bin = match spec.verb.as_str() {
            "egrep" => "/usr/bin/egrep",
            "fgrep" => "/usr/bin/fgrep",
            _ => "/usr/bin/grep",
        };
        let mut child = Command::new(bin)
            .args(&spec.fallback[1..])
            .current_dir(&spec.cwd)
            .env("LC_ALL", "C.UTF-8")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("system grep spawns");
        let mut stdin_pipe = child.stdin.take().expect("stdin piped");
        // Real grep may exit early (-m) and stop reading; a write error (EPIPE)
        // is then the OS-native producer death, not a test failure. Dropping
        // the handle delivers EOF for the full-scan case.
        let _ = stdin_pipe.write_all(stdin);
        drop(stdin_pipe);
        let out = child.wait_with_output().expect("system grep runs");
        (
            out.stdout,
            out.stderr,
            out.status.code().expect("grep exits normally"),
        )
    }

    /// The full command prefix before the served grep member — pipeline
    /// members plus any preceding `&&`/`||`/`;`-joined segments (all
    /// preserved verbatim in the rewrite); `real_run_shell` captures the
    /// stream from it. stdin-fed rows only.
    fn producer_command(command: &str) -> String {
        let segments = split_segments(command, ShellPlatform::Unix).expect("segments");
        let gidx = segments
            .iter()
            .position(|(s, _)| {
                let v = first_word(s);
                is_grep_verb(v, ShellPlatform::Unix)
                    || segment_contains_grep(s, v, ShellPlatform::Unix)
            })
            .expect("grep member");
        let mut start = gidx;
        // Walk back over any preceding segment (pipes, `&&`, `||`, `;`, blank
        // lines): the rewrite preserves them verbatim, so the prefix assertion
        // needs them in `out`. Rows with non-pipe segments before the producer
        // are self-consistent (both parity sides get the same bytes) — their
        // patterns must not match those segments' own output, which the
        // authentic pipeline's grep never sees.
        while start > 0
            && matches!(
                segments[start - 1].1.as_str(),
                "|" | "|&" | "&&" | "||" | ";" | "\n"
            )
        {
            start -= 1;
        }
        let mut out = String::new();
        for (i, (seg, conn)) in segments[start..gidx].iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(seg);
            if i + 1 < gidx - start {
                out.push(' ');
                out.push_str(conn);
            }
        }
        out
    }

    /// Rejoin `(segment, connector)` pairs from `start` on, prefixed by that
    /// segment's own connector — the verbatim-tail shape both tail helpers
    /// below share (the caller decides the anchor and early-return cases).
    fn rejoin_from(segments: &[(String, String)], start: usize) -> String {
        let mut out = format!(" {}", segments[start].1);
        for (seg, conn) in &segments[start + 1..] {
            out.push(' ');
            out.push_str(seg);
            if !conn.is_empty() {
                out.push(' ');
                out.push_str(conn);
            }
        }
        out
    }

    /// The original command's text from its first `|`/`|&` connector on — the
    /// tail that must survive the rewrite verbatim (never analyzed).
    fn pipeline_tail(command: &str) -> Option<String> {
        let segments = split_segments(command, ShellPlatform::Unix).ok()?;
        let first = segments
            .iter()
            .position(|(_, c)| matches!(c.as_str(), "|" | "|&"))?;
        Some(rejoin_from(&segments, first))
    }

    /// The joined twin of [`Analyzed`]: the rewrite as one command.
    type JoinedAnalyzeOutput = (
        Vec<EngineSpec>,
        Vec<(usize, String, bool)>,
        String,
        Vec<GrepOutcome>,
    );

    /// Analyze and join, exactly as the production caller does for the unix
    /// platform this parity lane runs under, so the assertions below read one
    /// command's rewrite. Nothing here executes a rewrite, so the executable
    /// path it names is a stand-in.
    fn analyze_joined(
        command: &str,
        ws: &Path,
        home: &Path,
        allow_single: bool,
    ) -> Result<JoinedAnalyzeOutput, AnalyzeFailure> {
        let analyzed = analyze_command(command, ws, home, ShellPlatform::Unix, allow_single)?;
        let jsons: Vec<String> = analyzed.specs.iter().map(spec_json).collect();
        let (rewritten, _) =
            join_rewritten(&analyzed.segments, &jsons, ShellPlatform::Unix, "/mahbot")
                .expect("the unix join never refuses");
        Ok((
            analyzed.specs,
            analyzed.shapes,
            rewritten,
            analyzed.outcomes,
        ))
    }

    /// Assert the engine's output/exit are byte-identical to the real grep.
    /// Recursive DIRECTORY walks (the parallel-walk path) may order files
    /// differently across workers, so those are compared as sorted line-sets;
    /// everything else stays byte-exact (the fast matcher's ordering is pinned).
    fn assert_parity(command: &str, ws: &Path, home: &Path) {
        let (specs, _, rewritten, _) = analyze_joined(command, ws, home, true)
            .unwrap_or_else(|e| panic!("{command}: expected servable, got {}", e.reason));
        assert_eq!(specs.len(), 1, "{command}: expected one grep member");
        if specs[0].stdin {
            assert_stdin_parity(command, &specs[0], &rewritten, ws, home);
            return;
        }
        let parallel = spec_uses_parallel_walk(&specs[0]);
        let (eout, eerr, ecode) = engine_run(&specs[0]);
        let piped = command.split_whitespace().any(|w| w == "|" || w == "|&");
        let (rout, rerr, rcode) = if piped {
            let tail = pipeline_tail(command).unwrap_or_default();
            assert!(
                !tail.is_empty() && rewritten.ends_with(&tail),
                "{command}: pipeline tail not preserved verbatim (rewritten: {rewritten})"
            );
            real_run_fallback(&specs[0])
        } else {
            real_run_shell(command, ws, home)
        };
        if parallel {
            assert_eq!(
                sorted_lines(&eout),
                sorted_lines(&rout),
                "stdout mismatch (parallel-walk ordering) for {command}"
            );
            assert_eq!(
                sorted_lines(&eerr),
                sorted_lines(&rerr),
                "stderr mismatch (parallel-walk ordering) for {command}"
            );
        } else {
            assert_eq!(eout, rout, "stdout mismatch for {command}");
            assert_eq!(eerr, rerr, "stderr mismatch for {command}");
        }
        assert_eq!(ecode, rcode, "exit mismatch for {command}");
    }

    /// Records (separated by `\n` or BSD's `--null`/`-z` `\0` terminator)
    /// sorted byte-wise; ordering-insensitive parity for parallel-walk rows.
    fn sorted_lines(bytes: &[u8]) -> Vec<&[u8]> {
        let mut v: Vec<&[u8]> = bytes
            .split(|&b| b == b'\n' || b == b'\0')
            .filter(|l| !l.is_empty())
            .collect();
        v.sort();
        v
    }

    /// The members after the served grep member, rejoined with their
    /// connectors — must survive the rewrite verbatim (stdin-fed rows).
    fn grep_tail(command: &str) -> Option<String> {
        let segments = split_segments(command, ShellPlatform::Unix).ok()?;
        let gidx = segments.iter().position(|(s, _)| {
            let v = first_word(s);
            is_grep_verb(v, ShellPlatform::Unix) || segment_contains_grep(s, v, ShellPlatform::Unix)
        })?;
        if gidx + 1 >= segments.len() {
            return Some(String::new());
        }
        Some(rejoin_from(&segments, gidx))
    }

    /// stdin-fed parity: capture the producer's stream through a real shell,
    /// then feed the same bytes to the engine and the real grep.
    fn assert_stdin_parity(
        command: &str,
        spec: &EngineSpec,
        rewritten: &str,
        ws: &Path,
        home: &Path,
    ) {
        let piped = command.split_whitespace().any(|w| w == "|" || w == "|&");
        assert!(piped, "{command}: stdin serve requires a pipeline");
        let producer = producer_command(command);
        assert!(
            !producer.is_empty() && rewritten.starts_with(&producer),
            "{command}: producer not preserved verbatim (rewritten: {rewritten})"
        );
        let tail = grep_tail(command).unwrap_or_default();
        assert!(
            tail.is_empty() || rewritten.ends_with(&tail),
            "{command}: pipeline tail not preserved verbatim (rewritten: {rewritten})"
        );
        let (stream, _, _) = real_run_shell(&producer, ws, home);
        let (eout, eerr, ecode, stream_bytes) = engine_run_with_stdin(spec, &stream);
        let (rout, rerr, rcode) = real_run_fallback_with_stdin(spec, &stream);
        // Pins the stream-size marker chain: report_stream_bytes stdin serves
        // must emit it. The count is the searcher's bytes through the
        // CountingReader — the head is re-read through it, so an EOF read
        // reports the exact stream length; only -m/-l early stops report a
        // bounded prefix.
        if spec.report_stream_bytes {
            match stream_bytes {
                Some(n) if stream.len() <= BINARY_WINDOW => {
                    assert_eq!(n, stream.len() as u64, "stream-size marker for {command}");
                }
                Some(n) => assert!(
                    n > 0 && n <= stream.len() as u64,
                    "stream-size marker out of range for {command}"
                ),
                None => panic!("stream-size marker missing for {command}"),
            }
        } else {
            assert_eq!(stream_bytes, None, "marker suppressed for {command}");
        }
        assert_eq!(eout, rout, "stdout mismatch for {command}");
        assert_eq!(eerr, rerr, "stderr mismatch for {command}");
        assert_eq!(ecode, rcode, "exit mismatch for {command}");
    }

    /// A member whose output `sh` sends to a file: the engine's own stdout —
    /// which the rewrite hands to the same redirect — must be byte-identical to
    /// what the ORIGINAL command's shell left in `target`, and the other stream
    /// must reach both sides' capture identically. Output redirects only
    /// (`>`, `>>`, `&>`; a `2>` row's shell half is pinned by
    /// `unix_operator_pins`), and stderr-free rows, so a merged `&>` file is the
    /// member's stdout alone.
    fn assert_redirect_parity(command: &str, target: &str, ws: &Path, home: &Path) {
        let (specs, _, rewritten, _) = analyze_joined(command, ws, home, false)
            .unwrap_or_else(|e| panic!("{command}: expected servable, got {}", e.reason));
        assert_eq!(specs.len(), 1, "{command}: expected one grep member");
        assert!(
            rewritten.ends_with(target),
            "{command}: the redirect rides the rewrite verbatim: {rewritten}"
        );
        let (eout, eerr, ecode) = engine_run(&specs[0]);
        let path = ws.join(target);
        let _ = fs::remove_file(&path);
        let (rout, rerr, rcode) = real_run_shell(command, ws, home);
        assert!(
            rout.is_empty(),
            "{command}: the original's stdout went to {target}"
        );
        let written = fs::read(&path)
            .unwrap_or_else(|e| panic!("{command}: the original's shell wrote {target}: {e}"));
        if spec_uses_parallel_walk(&specs[0]) {
            assert_eq!(
                sorted_lines(&eout),
                sorted_lines(&written),
                "{command}: the engine's stream must be what the original wrote to {target}"
            );
        } else {
            assert_eq!(
                eout, written,
                "{command}: the engine's stream must be what the original wrote to {target}"
            );
        }
        assert_eq!(eerr, rerr, "{command}: stderr");
        assert_eq!(ecode, rcode, "{command}: exit");
    }

    /// Assert the command falls back (original command untouched).
    fn assert_falls_back(command: &str, ws: &Path, home: &Path) {
        assert!(
            analyze_joined(command, ws, home, false).is_err(),
            "{command}: expected fallback, got served"
        );
    }

    /// Assert the command falls back with a specific reason (pins the label
    /// each row falls back with).
    fn assert_falls_back_reason(command: &str, ws: &Path, home: &Path, reason: &str) {
        let err = analyze_joined(command, ws, home, false)
            .err()
            .unwrap_or_else(|| panic!("{command}: expected fallback, got served"));
        assert_eq!(err.reason.to_string(), reason, "{command}");
    }

    #[test]
    #[expect(clippy::too_many_lines)] // differential parity matrix
    fn differential_parity_matrix() {
        let (_tmp, ws, home) = fixture();

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
            // ── Producer-first pipelines: the first grep in a pipeline is
            //    served from the producer's stdin (BSD stdin semantics) ──
            "cat a.txt | grep foo",
            "cat a.txt | grep -n foo | head -2",
            "cat a.txt | grep -c foo",
            "cat a.txt | grep -l foo",
            "cat a.txt | grep -H foo",
            "cat a.txt | grep -m1 foo",
            "cat a.txt | grep -l --null foo",
            "cat a.txt | grep -o foo | head -3",
            "cat a.txt | grep -i FOO",
            "cat a.txt | grep -v bar",
            "cat a.txt | grep -cl foo",
            "cat ctx.txt | grep -A1 -B1 d",
            "cat a.txt | egrep 'foo|bar'",
            "cat a.txt | fgrep foo",
            "cat a.txt | grep foo | grep oo | head -3", // grep-on-grep: second preserved verbatim
            "seq 1 100 | grep 5 | head -5",
            "seq 1 100000 | grep -m1 5", // -m early stop on a >32 KiB stream (bounded marker branch)
            "cat bindir/bin1.dat | grep needle", // binary stdin: NUL in the window
            "echo hi | grep x",          // no-match stdin: exit 1, empty output
            "cd sub && cat s.txt | grep needle", // cd-prefixed producer (stream captured from sub)
            "cd sub && echo hi && cat s.txt | grep needle", // non-cd && segment before the producer
            "false || cat a.txt | grep foo", // ||-joined producer (walk-back covers ||)
            "cat a.txt | grep foo 2>&1 | wc -l", // member-side stderr merge: marker suppressed
            // ── Raw-token operands: quoted/escaped globs and ~ are literal ──
            "grep -rn needle '~'",
            "grep -rn needle '*.txt'",
            "grep -rn needle \\*.txt",
            "grep -rn needle 'a\"b.txt'",
            "grep -rn needle '~/htree'", // literal path, no such file (stderr row)
            "grep -rn needle '~user'",   // literal path, no such file (stderr row)
            "grep -rn needle ~/\"htree\"", // mixed form: `~/` expands, quoted rest joins
            // ── any-length pipelines: any tail is served verbatim (grep
            //    member parity; full-pipeline parity lives in the e2e bench).
            //    The 2>&1 row covers the benign ordering only — engine stderr
            //    is buffered to finish while BSD emits at operand-open, so
            //    missing-first or order-sensitive tails diverge (accepted
            //    residual). Tails are never analyzed: second greps, grep
            //    introducers (xargs grep) and 3+ members are preserved ──
            "grep -rn x plain | wc -l",
            "grep -rn x plain | tail -2",
            "grep -rn x plain | sort",
            "grep -rn x plain missing.txt 2>&1 | wc -l",
            "grep -rn x plain | grep -v 'plain/d1' | head -5", // 3+ member, second grep preserved
            "grep foo a.txt | grep oo", // 2-member grep|grep (tail grep preserved)
            "grep foo a.txt | xargs grep oo", // grep introducer preserved in the tail
            "grep foo a.txt | head -1 | wc -l", // 3+ member, single-file first grep
            // Quoted `~` + unquoted glob is a literal cwd-relative path
            // (`"~"/*.txt` must not home-strip the tilde).
            "grep -rn needle 'sub/~'/*.txt",
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

    /// The glued spellings, observed end-to-end: the served rewrite's own stream
    /// lands in the file the command glued the operator to (the shell performs
    /// the write), and a backgrounded search keeps what follows its `&`.
    #[test]
    fn glued_operator_and_background_parity() {
        let (_tmp, ws, home) = fixture();

        // A glued target (`>file`, `>>file`, `&>file`), the spaced spelling as
        // the control, and the recursive shape the agents actually issue.
        for (command, target) in [
            (
                "grep -n x a.txt b.txt>redirect-glued.txt",
                "redirect-glued.txt",
            ),
            (
                "grep -n x a.txt b.txt>>redirect-glued-append.txt",
                "redirect-glued-append.txt",
            ),
            (
                "grep -n x a.txt b.txt&>redirect-glued-both.txt",
                "redirect-glued-both.txt",
            ),
            (
                "grep -n x a.txt b.txt > redirect-closed.txt",
                "redirect-closed.txt",
            ),
            (
                "grep -rn x plain>redirect-glued-walk.txt",
                "redirect-glued-walk.txt",
            ),
        ] {
            assert_redirect_parity(command, target, &ws, &home);
        }

        // A backgrounded search: served, with everything after its `&` left to
        // the shell verbatim.
        for (command, tail) in [
            ("grep -rn x plain & wait", "& wait"),
            // The join spaces the `;` connector out, hence the tail's spelling.
            ("grep -rn x plain & wait; echo AFTER", "& wait ; echo AFTER"),
        ] {
            let (specs, _, rewritten, _) = analyze_joined(command, &ws, &home, false)
                .unwrap_or_else(|e| panic!("{command}: expected servable, got {}", e.reason));
            assert_eq!(specs.len(), 1, "{command}: the search is served");
            assert!(
                rewritten.ends_with(tail),
                "{command}: the tail after `&` survives verbatim: {rewritten}"
            );
        }
        assert_parity("grep -rn x plain & wait", &ws, &home);

        // A comment takes the rest of its line with it: the served search is the
        // separated spelling's word set (nothing the comment names reaches
        // grep's argv) and the original's own `sh` runs nothing after the `#`,
        // so the two streams still have to agree byte for byte.
        assert_parity("grep -rn x plain # note & echo TAIL-RAN", &ws, &home);
    }

    #[test]
    fn stdin_m_early_stop_stops_consuming() {
        // `seq 1 100000 | grep -m1 5`: the match sits inside the 32 KiB head,
        // so a -m1 serve must stop reading once found (BSD instant-exit). The
        // marker reports consumed bytes — well short of the full stream; a
        // full scan would consume (and report) all of it.
        let (_tmp, ws, home) = fixture();
        let (specs, _, _, _) =
            analyze_joined("seq 1 100000 | grep -m1 5", &ws, &home, true).expect("servable");
        let (stream, _, _) = real_run_shell("seq 1 100000", &ws, &home);
        assert!(stream.len() > BINARY_WINDOW, "fixture stream too small");
        let (_, _, _, stream_bytes) = engine_run_with_stdin(&specs[0], &stream);
        let n = stream_bytes.expect("marker emitted");
        assert!(
            n > 0 && n < stream.len() as u64,
            "consumed {n} of {}: expected early stop",
            stream.len()
        );
    }

    #[test]
    fn fallback_triggers() {
        let (_tmp, ws, home) = fixture();
        let rows: &[&str] = &[
            "grep x",                                  // stdin (first member: tty-hang protection)
            "grep x -",                                // - operand (first member)
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
            "grep -rn needle '~'", // quoted ~ is literal → single file → perf gate
            "cat <<EOF\nfoo\nEOF\ngrep x a.txt b.txt", // heredoc feeds a non-grep member
            "git grep x",          // git grep
            "if grep x a.txt; then echo hi; fi", // compound
            "for x in a; do grep x a.txt; done", // compound
            "( grep x a.txt b.txt )", // subshell group
            "{ grep x a.txt; }",   // brace group
            "(cd sub && grep x a.txt b.txt)", // subshell cd + group
            "xargs grep x",        // indirect
            "sh -c 'grep x a.txt'", // indirect
            "cd $HOME && grep x a.txt", // cd untrackable
            "cd - && grep x a.txt", // cd $OLDPWD
            "! grep x a.txt",      // negation
            "grep x a.txt |",      // trailing pipe
            "sudo grep x a.txt",   // env prefix
            "grep -w 'foo\\|bar' a.txt b.txt", // -w + alternation (word_safe)
            "echo $(echo \\) ; grep x a.txt", // unterminated $(...) — escape-aware span stays open
            "echo hello && echo world", // no grep at all
            // Producer-first stdin rejects: structural, not producer-based.
            "cat a.txt | sudo grep foo",  // nested introducer producer
            "cat a.txt | xargs grep foo", // nested introducer producer
        ];
        for row in rows {
            assert_falls_back(row, &ws, &home);
        }
        // Pinned reason labels: `;;` is a case terminator (compound, not an
        // empty member); empty pipeline members stay rejected; non-grep
        // pipelines are "no grep", not "pipeline shape".
        assert_falls_back_reason(
            "case $x in a) grep x a.txt;; esac",
            &ws,
            &home,
            "nested grep",
        );
        assert_falls_back_reason(
            "grep x a.txt | | wc -l",
            &ws,
            &home,
            "empty command or pipeline member",
        );
        assert_falls_back_reason("cat f | head", &ws, &home, "no grep");
        // Producer-first stdin rejects: structural classes, not producer-based.
        assert_falls_back_reason(
            "cat a.txt | grep foo b.txt",
            &ws,
            &home,
            "stdin with operands",
        );
        assert_falls_back_reason("cat a.txt | grep foo -", &ws, &home, "stdin with operands");
        assert_falls_back_reason("printf 'x' | grep -r foo", &ws, &home, "stdin with -r");
        // Any-length pipelines with a single-file first grep fall back for the
        // perf gate, not the pipeline shape (their tails are preserved).
        assert_falls_back_reason("grep x a.txt | head -1 | wc -l", &ws, &home, "single file");
        assert_falls_back_reason("grep x a.txt | grep y", &ws, &home, "single file");
        assert_falls_back_reason("grep x a.txt | xargs grep y", &ws, &home, "single file");
    }

    #[test]
    fn per_segment_skip_with_served_sibling() {
        // A single-file grep (perf gate) does not poison a sibling servable
        // recursive grep: the unservable member is kept verbatim, the servable
        // one rewritten, and the whole command is served.
        let (_tmp, ws, home) = fixture();
        let (specs, _, rewritten, outcomes) =
            analyze_joined("grep x a.txt; grep -rn needle sub", &ws, &home, false)
                .expect("chain with a servable sibling served");
        assert_eq!(specs.len(), 1, "only the recursive grep is served");
        assert!(
            rewritten.starts_with("grep x a.txt ;"),
            "first grep kept verbatim: {rewritten}"
        );
        assert!(
            rewritten.contains("__grep-engine"),
            "rewritten has engine verb: {rewritten}"
        );
        assert_eq!(outcomes.len(), 2, "one skip + one serve recorded");
        assert!(!outcomes[0].served && outcomes[0].reason == "single file");
        assert!(outcomes[1].served && outcomes[1].reason.is_empty());

        // A compound (`if`) containing a grep is skipped (kept verbatim) while
        // a sibling recursive grep is served.
        let (specs, _, rewritten, outcomes) = analyze_joined(
            "grep -rn needle sub; if grep x a.txt; then echo hi; fi",
            &ws,
            &home,
            false,
        )
        .expect("compound sibling does not poison");
        assert_eq!(specs.len(), 1, "only the recursive grep is served");
        assert!(
            rewritten.contains("__grep-engine"),
            "recursive grep served: {rewritten}"
        );
        assert!(
            rewritten.ends_with("if grep x a.txt ; then echo hi ; fi"),
            "compound kept verbatim: {rewritten}"
        );
        assert_eq!(
            outcomes.len(),
            2,
            "serve + the compound segment containing grep; bare `then`/`fi` carry no grep"
        );
        assert!(outcomes[0].served && outcomes[0].reason.is_empty());
        assert!(!outcomes[1].served && outcomes[1].reason == "nested grep");
    }

    #[test]
    fn substitution_escape_resegments() {
        // Escape-aware substitution scans (consume_substitution) change
        // segmentation: an escaped backtick no longer truncates a backtick
        // span, so the trailing grep is a real separate member and gets served
        // (the old scan swallowed it into an unterminated span → fallback).
        // The serve→fallback direction of the same fix is a fallback_triggers
        // row (`echo $(echo \) ; grep x a.txt`).
        let (_tmp, ws, home) = fixture();
        analyze_joined("echo `a\\`b` ; grep x a.txt b.txt", &ws, &home, true)
            .expect("escaped backtick: trailing grep is a separate served member");
    }

    #[test]
    fn exclusion_delta_is_rg_default() {
        // An approved behavioral delta: recursive walks skip
        // hidden/gitignored content; explicit file operands always searched.
        let (_tmp, ws, home) = fixture();
        let (specs, _, _, _) =
            analyze_joined("grep -r x ign", &ws, &home, true).expect("ign walk servable");
        let (eout, _, ecode) = engine_run(&specs[0]);
        let text = String::from_utf8_lossy(&eout).to_string();
        assert_eq!(ecode, 0);
        assert!(text.contains("ign/visible.txt"), "visible searched: {text}");
        assert!(!text.contains(".hidden.txt"), "hidden skipped: {text}");
        assert!(!text.contains("skip.log"), "gitignored skipped: {text}");

        // Explicit file operands bypass the exclusion filters entirely.
        let (specs, _, _, _) =
            analyze_joined("grep -r x ign/.hidden.txt ign/skip.log", &ws, &home, true)
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
        let (_tmp, ws, home) = fixture();
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

// ── Redirect-consumption pins (non-gated: pure parsing) ───────────────
// The token classifier is shared with the read-only guard (readonly.rs);
// these rows pin the TokenKind→(redirect, needs_target) mapping for the
// divergent redirect shapes so a wrong mapping fails loudly here.

#[cfg(test)]
mod redirect_token_pins {
    use super::*;

    fn redirects_of(segment: &str) -> (Vec<String>, Vec<String>) {
        let mut words = grep_tokenize(segment, ShellPlatform::Unix).expect("tokenize");
        if words.first().is_some_and(|w| w.value == "grep") {
            words.remove(0);
        }
        let parsed = parse_grep_words(&words, "grep", ShellPlatform::Unix).expect("parse");
        (parsed.redirects, parsed.operand_tokens)
    }

    #[test]
    fn divergent_redirect_shapes_consume_targets_like_the_guard() {
        // `10>` (multi-digit fd) expects a target word; `1>0` (fd + target
        // merged) does not — the following word stays a grep operand.
        let (redirects, operands) = redirects_of("grep foo a.txt 10> out.txt");
        assert_eq!(redirects, ["10>", "out.txt"], "10> must consume its target");
        assert_eq!(operands, ["a.txt"]);

        let (redirects, operands) = redirects_of("grep foo a.txt 1>0 out.txt");
        assert_eq!(redirects, ["1>0"], "1>0 is self-contained");
        assert_eq!(operands, ["a.txt", "out.txt"], "out.txt stays an operand");
    }
}

// ── cd-target scan pins (non-gated: pure parsing) ────────────────────
// The cd option-skip/target-extract loop is shared with the read-only guard
// (readonly.rs); these rows pin the three-state scan and the grep engine's
// fail-closed mapping so a divergence fails loudly here. The guard's own
// fail-closed arm (bare / invalid option / extra operand → reset) is pinned
// by its cd tests (cd_flag_forms_fail_closed, cd_extra_operands_fail_closed).

#[cfg(test)]
mod cd_scan_pins {
    use super::super::scan::CdScan;
    use super::*;

    /// Scan the words of a `cd` segment (verb excluded) via the shared helper.
    fn scan(segment: &str) -> CdScan<'_> {
        let words: Vec<&str> = segment.split_whitespace().collect();
        super::super::scan::cd_target_after_options(&words, 0)
    }

    #[test]
    fn cd_option_grammar_stays_shared_and_fail_closed() {
        // (words after the verb, expected scan). Target rows carry the
        // quote-stripped target and the index after it.
        let rows: &[(&str, CdScan<'static>)] = &[
            ("sub", CdScan::Target("sub", 1)),
            ("-P /tmp", CdScan::Target("/tmp", 2)),
            ("-L /tmp", CdScan::Target("/tmp", 2)),
            ("-PL /tmp", CdScan::Target("/tmp", 2)),
            ("-- -P", CdScan::Target("-P", 2)), // `--` ends option parsing
            ("--", CdScan::Bare),
            ("-P", CdScan::Bare),           // flag-only → $HOME
            ("", CdScan::Bare),             // bare cd
            ("-", CdScan::Target("-", 1)),  // $OLDPWD target, never an option
            ("-e /tmp", CdScan::BadOption), // invalid option errors at runtime
            ("-Pe /tmp", CdScan::BadOption),
            ("\"-P\"", CdScan::Target("-P", 1)), // quoted flag is a literal target
            ("\"/tmp\"", CdScan::Target("/tmp", 1)), // target is quote-stripped
        ];
        for (input, expected) in rows {
            assert_eq!(&scan(input), expected, "cd {input}");
        }

        // The grep engine's fail-closed mapping of the same three states:
        // bare → $HOME; invalid option / extra operand → fallback.
        let tmp = tempfile::tempdir().expect("tempdir");
        let ws = tmp.path().join("ws");
        let home = tmp.path().join("home");
        fs::create_dir_all(&ws).expect("ws");
        fs::create_dir_all(&home).expect("home");
        assert_eq!(
            resolve_cd("cd", &ws, &home, ShellPlatform::Unix).expect("bare cd"),
            canonical_or_lexical(&home, ShellPlatform::Unix)
        );
        assert_eq!(
            resolve_cd("cd -P", &ws, &home, ShellPlatform::Unix).expect("flag-only cd"),
            canonical_or_lexical(&home, ShellPlatform::Unix)
        );
        for bad in ["cd -e sub", "cd -Pe sub", "cd sub extra", "cd -- -P extra"] {
            assert!(
                matches!(
                    resolve_cd(bad, &ws, &home, ShellPlatform::Unix),
                    Err(Fallback::CdUntrackable)
                ),
                "{bad}: expected CdUntrackable"
            );
        }
    }
}

// ── Windows fnmatch parity (host-differential) ───────────────────────────
// The Windows matcher is pure Rust (no libc on that platform), so the host's
// own `fnmatch` is the only oracle available for it. Both are the SAME
// semantics the unix lane uses (the engine's filters and the shell's glob
// expansion), so agreement here is what makes the Windows spelling trustworthy:
// the cross-product below, malformed spellings included, is this matcher's
// whole semantics pin.

#[cfg(all(test, unix))]
mod windows_fnmatch_parity {
    use super::*;

    /// This host's `libc::fnmatch`, called exactly as [`fnmatch_flags`] calls
    /// it on unix.
    fn host_fnmatch(pattern: &str, name: &str, period: bool) -> bool {
        let p = std::ffi::CString::new(pattern).expect("pattern has no NUL");
        let n = std::ffi::CString::new(name).expect("name has no NUL");
        let flags = if period { libc::FNM_PERIOD } else { 0 };
        // SAFETY: both CStrings are NUL-free and outlive the call.
        unsafe { libc::fnmatch(p.as_ptr(), n.as_ptr(), flags) == 0 }
    }

    #[test]
    fn windows_matcher_agrees_with_the_host() {
        let patterns = [
            "*", "?", "a*", "*a", "a?c", "*.txt", "*.*", "[abc]", "[abc]*", "[!abc]", "[^a]",
            "[a-z]*", "a[b-d]e", "[a-]", "[]a]", r"\*", r"a\?c", ".*", ".x", "*[!.]*", "*x*",
            // Malformed classes: an unterminated bracket and a reversed range,
            // which the engine's glob walk must treat exactly as the host does.
            "[abc", "a[", "[z-a]",
        ];
        let names = [
            "", "a", "ab", "abc", "a.txt", "ab.txt", ".a", ".a.txt", ".hidden", ".", "..", "*",
            "?", "\\", "a*c", "a?c", "a-b", "]", "a]", "ace", "axe", ".txt", "a/b", "abc ",
        ];
        for pattern in patterns {
            for name in names {
                for period in [false, true] {
                    assert_eq!(
                        windows::fnmatch(pattern, name, period),
                        host_fnmatch(pattern, name, period),
                        "fnmatch({pattern:?}, {name:?}, period={period})"
                    );
                }
            }
        }
    }
}

// ── Segmenter divergence pins (non-gated: pure parsing, both splitters) ──
// Both splitters run the shared core (shell::segment_command); these rows
// pin the two policies' deliberate divergence so a silent policy swap fails
// loudly. Unifying the policies is a separate decision.

#[cfg(test)]
mod segmenter_pins {
    use super::super::extract_command_segments;
    use super::*;

    #[test]
    fn divergent_policies_stay_pinned() {
        // (input, profile segments) — profile silently skips empty segments
        // and drops a backslash before ordinary chars.
        let profile_rows: &[(&str, &[&str])] = &[
            // Unquoted backslash before an ordinary char: dropped (kept only
            // before escape-sensitive chars).
            ("echo \\a", &["echo a"]),
            // `|&` is not a compound connector: the `&` starts its own segment.
            ("a |& b", &["a", "& b"]),
            // `;;` outside a case: two `;` separators, empty segment skipped.
            ("echo a ;; echo b", &["echo a", "echo b"]),
            // `sh`'s comment is ordinary text to the profile splitter, which
            // never runs a rewrite: the `&` inside it is not a separator either.
            ("grep x a.txt # c & echo A", &["grep x a.txt # c & echo A"]),
        ];
        for (input, expected) in profile_rows {
            let segs = extract_command_segments(input);
            let got: Vec<&str> = segs.iter().map(String::as_str).collect();
            assert_eq!(got, *expected, "profile: {input:?}");
        }

        // (input, grep (segment, connector) pairs) — grep errors on empty
        // members (except blank lines and `;;` case-arm terminators) and
        // always preserves backslashes.
        let grep_ok: &[(&str, &[(&str, &str)])] = &[
            // Backslash always preserved: dropping `\*` would turn the operand
            // into a shell glob.
            ("grep \\*.txt a.txt", &[("grep \\*.txt a.txt", "")]),
            // `|&` is a single compound connector (stderr merge).
            ("a |& grep x a.txt", &[("a", "|&"), ("grep x a.txt", "")]),
            // An unquoted `#` comments out the rest of its line only: neither
            // the `&` inside the comment nor the member after it survives as a
            // connector or a command, while the line below still does.
            (
                "grep x a.txt # c & echo A\nls -la",
                &[("grep x a.txt", "\n"), ("ls -la", "")],
            ),
        ];
        for (input, expected) in grep_ok {
            let got = split_segments(input, ShellPlatform::Unix).expect("expected segments");
            let got: Vec<(&str, &str)> =
                got.iter().map(|(s, c)| (s.as_str(), c.as_str())).collect();
            assert_eq!(got.as_slice(), *expected, "grep: {input:?}");
        }
        // `;;` outside a case: empty segment before `;` is a syntax error.
        assert!(
            split_segments("echo a ;; echo b", ShellPlatform::Unix).is_err(),
            "grep: ;; outside case"
        );
    }
}

// ── Display-path pins (non-gated: the unix reading must not drift) ──────
// The platform split in these helpers is what keeps a unix answer unix-shaped:
// `\` is an ordinary filename character there, and a walked entry is always
// `/`-joined.

#[cfg(test)]
mod platform_string_pins {
    use super::*;

    #[test]
    fn unix_display_paths_and_filters_read_only_slash() {
        // A walked entry joins on `/`, including the degenerate empty operand
        // spelling (which renders a leading separator).
        assert_eq!(
            traversal_display(
                "/ws/src",
                Path::new("/ws/src"),
                Path::new("/ws/src/main.rs"),
                ShellPlatform::Unix
            ),
            "/ws/src/main.rs"
        );
        assert_eq!(
            traversal_display(
                "",
                Path::new("/ws"),
                Path::new("/ws/main.rs"),
                ShellPlatform::Unix
            ),
            "/main.rs"
        );
        // Only a trailing `/` is trimmed; a trailing `\` is part of the name.
        assert_eq!(
            traversal_display(
                r"/ws\dir",
                Path::new("/ws/dir"),
                Path::new("/ws/dir/a.rs"),
                ShellPlatform::Unix
            ),
            r"/ws\dir/a.rs"
        );

        // A basename splits on `/` alone: a filename may contain a backslash.
        assert_eq!(display_basename("a/b/c.rs", ShellPlatform::Unix), "c.rs");
        assert_eq!(display_basename(r"a\b", ShellPlatform::Unix), r"a\b");
        assert_eq!(display_basename(r"a\b", ShellPlatform::Windows), "b");
        assert_eq!(display_basename("a/b", ShellPlatform::Windows), "b");
    }

    /// A `\r` is dropped only as the `\n`'s own carriage return: a line whose
    /// last byte is a bare `\r` (a file with no final newline) keeps it on both
    /// platforms, exactly as the host grep delivers it.
    #[test]
    fn windows_strips_the_carriage_return_only_with_its_newline() {
        assert_eq!(
            trim_line_terminator(b"line\r\n", ShellPlatform::Windows),
            b"line"
        );
        assert_eq!(
            trim_line_terminator(b"line\n", ShellPlatform::Windows),
            b"line"
        );
        assert_eq!(
            trim_line_terminator(b"line\r", ShellPlatform::Windows),
            b"line\r"
        );
        assert_eq!(
            trim_line_terminator(b"line\r", ShellPlatform::Unix),
            b"line\r"
        );
    }

    /// One canonicalization for both sides of the cwd gate: on Windows the
    /// verbatim (`\\?\`) prefix `fs::canonicalize` may add is stripped, so
    /// tracked and displayed paths stay in the platform's natural spelling, and
    /// a path that cannot be canonicalized falls back to its lexical spelling
    /// rather than failing the analysis.
    #[test]
    fn canonical_or_lexical_strips_the_windows_verbatim_prefix() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let canonical = canonical_or_lexical(tmp.path(), ShellPlatform::Windows);
        assert!(
            !canonical.to_string_lossy().starts_with(r"\\?\"),
            "{canonical:?}"
        );
        let missing = tmp.path().join("no-such-dir");
        assert!(
            canonical_or_lexical(&missing, ShellPlatform::Windows)
                .to_string_lossy()
                .contains("no-such-dir")
        );
    }
}

// ── Hand-off pins (non-gated: the rewrite's one machine-dependent part) ──
// Both hand-offs are driven from this host: the spec rides on the command line
// on unix and through a scratch file on Windows, and the Windows executable
// quoting is the one part of a rewrite that depends on where the process was
// installed.

#[cfg(test)]
mod handoff_pins {
    use super::super::SpecFiles;
    use super::*;

    /// The Windows rewrite quotes the executable in place, and cmd.exe keeps
    /// every metacharacter literal inside that quoting — an installation under
    /// `C:\Program Files (x86)\…` is served, not disabled machine-wide. A path
    /// carrying what cmd.exe rewrites even inside quotes is refused instead of
    /// written out as a line cmd would read differently.
    #[test]
    fn windows_hand_off_quotes_the_executable_and_refuses_only_what_cmd_rewrites() {
        let spec = render_only_json();
        let mut files = Vec::new();
        let fragment = render_served(
            &spec,
            &[],
            ShellPlatform::Windows,
            r"C:\Program Files (x86)\MahBot\mahbot.exe",
            &mut files,
        )
        .expect("a programme path cmd.exe reads literally inside quotes");
        assert!(
            fragment.starts_with(r#""C:\Program Files (x86)\MahBot\mahbot.exe" "#),
            "{fragment}"
        );
        assert!(fragment.contains(windows::SPEC_FILE_FLAG), "{fragment}");
        assert_eq!(files.len(), 1, "the spec rides a scratch file");
        assert!(files[0].exists(), "{:?}", files[0]);
        let file = files[0].clone();
        drop(SpecFiles(files));
        assert!(!file.exists(), "the call's guard removes it: {file:?}");

        for bad in [r"C:\Users\a%b\mahbot.exe", r"C:\Users\a!b\mahbot.exe"] {
            let err = render_served(&spec, &[], ShellPlatform::Windows, bad, &mut Vec::new())
                .expect_err("a path cmd.exe rewrites even inside quotes");
            assert!(err.to_string().contains("hand-off refused"), "{err}");
        }
    }

    /// On unix the executable's path is single-quoted for `sh` and the spec
    /// rides the command line.
    #[test]
    fn unix_hand_off_keeps_the_argv_transport() {
        let mut files = Vec::new();
        let fragment = render_served(
            &render_only_json(),
            &["2>&1".into()],
            ShellPlatform::Unix,
            "/opt/it's here/mahbot",
            &mut files,
        )
        .expect("unix hand-off");
        assert!(files.is_empty(), "no scratch file on unix");
        assert!(
            fragment.starts_with("'/opt/it'\\''s here/mahbot' "),
            "{fragment}"
        );
        assert!(fragment.contains(ENGINE_VERB), "{fragment}");
        assert!(
            fragment.ends_with(" 2>&1"),
            "redirects stay verbatim: {fragment}"
        );
    }

    /// The scratch-file transport's read half: the flag arm of [`read_spec`]
    /// takes the file's contents (the flag itself is never parsed as JSON), the
    /// argv arm still passes a bare spec straight through, and a path with
    /// nothing behind it is an error — the member the engine refuses, and the
    /// parent then refuses the call on.
    ///
    /// Driven at [`read_spec`] rather than through `run_engine`, which is not
    /// callable from this lane at all: it puts the process-wide SIGPIPE
    /// disposition back to its default, which this binary shares with tests
    /// whose producer writes into a child pipe expecting `EPIPE` (a sibling
    /// thread inside that window would be killed by SIGPIPE and take the whole
    /// test run down with it), and on a cannot-serve path it replaces this
    /// process with the real `grep`. Both forms reach `serve` as the same JSON
    /// string, which the parity tests drive directly; the end-to-end transport
    /// is covered by the manual smoke run of the built binary instead.
    #[test]
    fn the_spec_file_flag_reads_the_scratch_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("spec.json");
        fs::write(&path, r#"{"version":1}"#).expect("spec file");
        let file_args = |path: &Path| {
            vec![
                windows::SPEC_FILE_FLAG.to_string(),
                path.to_string_lossy().into_owned(),
                "grep".to_string(),
            ]
        };

        assert_eq!(
            read_spec(&file_args(&path)).expect("the flag reads the file"),
            r#"{"version":1}"#
        );
        assert!(
            read_spec(&file_args(&tmp.path().join("no-such-spec.json"))).is_err(),
            "an unreadable spec file is an error, never a parse of the flag"
        );
        assert_eq!(
            read_spec(&[r#"{"version":2}"#.to_string()]).expect("the argv arm passes it through"),
            r#"{"version":2}"#
        );
    }
}

// ── Read-only serve pins (no feature gate) ──────────────────────────────
// Both platforms are driven from this host: the platform is a value, and the
// guard reads it from its own context. The unix rows pin the argv hand-off and
// the single-file perf gate; the Windows rows pin the scratch-file hand-off.

// One gate for the whole harness: the pin modules below (and the platform
// readers they drive) share these fixtures, re-exported into file scope so each
// row names them directly.
#[cfg(test)]
use self::test_support::{render_only_json, serve_fixture, served, spec_cwd};

#[cfg(test)]
mod test_support {
    use super::*;

    /// The temp tree the platform pin modules serve against: a workspace holding
    /// the operand paths their rows name, and a home tree beside it. Bind the
    /// returned `TempDir` for the whole test — dropping it deletes the trees.
    pub(super) fn serve_fixture() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let ws = tmp.path().join("ws");
        let home = tmp.path().join("home");
        fs::create_dir_all(ws.join("src")).expect("ws/src");
        fs::create_dir_all(&home).expect("home");
        fs::write(ws.join("f.txt"), "needle\n").expect("f.txt");
        fs::write(ws.join("src/main.rs"), "fn needle() {}\n").expect("main.rs");
        (tmp, ws, home)
    }

    /// Serve `command` on `platform` with the engine probe bypassed and require a
    /// rewrite with no refusal — the one "this command is served" primitive both
    /// platform pin modules build on (they add the guard check and the refusal
    /// inversion respectively). Returns the rewrite and the scratch spec files,
    /// which the caller owns through [`SpecFiles`].
    pub(super) fn served(
        command: &str,
        ws: &Path,
        home: &Path,
        platform: ShellPlatform,
    ) -> (String, Vec<PathBuf>) {
        let mut serve = serve_command(command, ws, Some(home), platform, || true);
        assert!(
            serve.refusal.is_none(),
            "{command}: served, not refused: {:?}",
            serve.refusal
        );
        let rewritten = serve
            .rewritten
            .take()
            .unwrap_or_else(|| panic!("{command}: expected a rewrite"));
        (rewritten, serve.spec_files)
    }

    /// Run the engine in-process on one spec; returns (stdout, stderr, code).
    /// stdin-fed specs get an empty reader — never the test process's stdin.
    /// `run_engine` (the child-side entry) is deliberately not used: it puts the
    /// process-wide SIGPIPE disposition back to its default, which this binary
    /// shares with tests whose producer writes into a child pipe expecting
    /// `EPIPE`, and on a cannot-serve path it replaces the process with the real
    /// `grep`. Unix-gated with its rows: the Windows lane's reading is argued in
    /// `windows`, never run in-process here.
    #[cfg(unix)]
    pub(super) fn engine_run(spec: &EngineSpec) -> (Vec<u8>, Vec<u8>, i32) {
        let (out, err, code, _) = engine_run_with_stdin(spec, &[]);
        (out, err, code)
    }

    /// Run the engine in-process feeding explicit stdin bytes (stdin-fed rows).
    /// Returns (stdout, stderr, code, stripped stream-byte count): the last
    /// pins the stream-size marker chain (None when no marker was emitted).
    #[cfg(unix)]
    pub(super) fn engine_run_with_stdin(
        spec: &EngineSpec,
        stdin: &[u8],
    ) -> (Vec<u8>, Vec<u8>, i32, Option<u64>) {
        let matcher =
            build_matcher(&spec.patterns, spec.mode, &spec.flags).expect("matcher builds");
        let mut out = Output::new(OutputSink::Buffer(Vec::new()), output_limit(spec));
        let consumed = std::cell::Cell::new(false);
        let code = serve_into(spec, &matcher, &mut out, io::Cursor::new(stdin), &consumed);
        let Output {
            sink: OutputSink::Buffer(buf),
            mut err,
            ..
        } = out
        else {
            unreachable!("buffered sink")
        };
        // The parent strips the stream-size marker from stderr; mirror that.
        let stream_bytes = strip_stream_size_marker(&mut err);
        (buf, err, code, stream_bytes)
    }

    /// A spec whose rendering needs no operand on disk — the fixture for the
    /// hand-off and guard pins that drive [`render_served`] with an executable
    /// path of their own. Returned serialized, the form the renderer takes.
    pub(super) fn render_only_json() -> String {
        spec_json(&EngineSpec {
            version: PROTOCOL_VERSION,
            verb: "grep".into(),
            mode: MatchMode::Basic,
            flags: GrepFlags::default(),
            filters: Vec::new(),
            exclude_dir: Vec::new(),
            patterns: vec!["needle".into()],
            operands: vec![Operand {
                display: ".".into(),
                resolved: "/ws".into(),
                trailing_slash: false,
            }],
            cwd: "/ws".into(),
            fallback: vec!["grep".into()],
            piped: false,
            stdin: false,
            report_stream_bytes: false,
        })
    }

    /// The tracked cwd recorded in a Windows rewrite's scratch spec — the cwd
    /// the engine's own gate compares against. `files` is the rewrite's
    /// spec-file list.
    pub(super) fn spec_cwd(files: &[PathBuf]) -> String {
        let path = files.first().expect("the rewrite scratched a spec file");
        let json = fs::read_to_string(path).expect("spec file readable");
        let spec: EngineSpec = serde_json::from_str(&json).expect("spec parses");
        spec.cwd
    }
}

#[cfg(test)]
mod read_only_serve_pins {
    use super::super::SpecFiles;
    use super::super::readonly::{CheckContext, check_command};
    use super::*;

    /// Serve `command` on `platform` and assert the read-only guard accepts the
    /// rewrite. Returns the rewrite and the scratch spec files the caller must
    /// drop.
    fn served_and_guard_accepted(
        command: &str,
        ws: &Path,
        home: &Path,
        platform: ShellPlatform,
    ) -> (String, Vec<PathBuf>) {
        let (rewritten, files) = served(command, ws, home, platform);
        assert!(rewritten.contains(ENGINE_VERB), "{rewritten}");
        let ctx = CheckContext::for_platform(ws, platform);
        assert!(
            check_command(&rewritten, &ctx).is_ok(),
            "read-only guard rejected the {platform:?} engine rewrite: {rewritten}"
        );
        (rewritten, files)
    }

    /// Pins the module-header invariant: a generated rewrite must pass the
    /// read-only guard, whichever hand-off the platform uses.
    #[test]
    fn generated_rewrite_passes_the_read_only_guard() {
        let (_tmp, ws, home) = serve_fixture();

        // Unix: the spec rides on the command line as a single-quoted argv.
        let (rewritten, files) =
            served_and_guard_accepted("grep -rn needle .", &ws, &home, ShellPlatform::Unix);
        assert!(
            rewritten.starts_with('\''),
            "rewrite quotes the executable path: {rewritten}"
        );
        assert!(files.is_empty(), "unix serves the spec by argv");

        // The operator shapes the unix word reading hands back to the shell must
        // clear the guard too: a rewrite it rejects is silently dropped and the
        // read-only lane would run the real `grep` instead.
        for (command, tail) in [
            ("grep -rn needle . & echo done", "& echo done"),
            ("grep -rn needle . &", " &"),
            ("grep -rn needle<f.txt .", "<f.txt"),
        ] {
            let (rewritten, files) =
                served_and_guard_accepted(command, &ws, &home, ShellPlatform::Unix);
            assert!(
                rewritten.ends_with(tail),
                "{command}: the operator rides the rewrite: {rewritten}"
            );
            drop(SpecFiles(files));
        }

        // Windows: cmd.exe cannot carry the spec, so the rewrite names a
        // scratch file it must be able to read.
        let (rewritten, files) = served_and_guard_accepted(
            "grep -n needle src/main.rs",
            &ws,
            &home,
            ShellPlatform::Windows,
        );
        assert_eq!(files.len(), 1, "one spec file per served member");
        assert!(rewritten.contains(windows::SPEC_FILE_FLAG), "{rewritten}");
        let json = fs::read_to_string(&files[0]).expect("spec file readable");
        let spec: serde_json::Value = serde_json::from_str(&json).expect("spec parses as JSON");
        assert_eq!(spec["verb"], "grep");
        let file = files[0].clone();
        drop(SpecFiles(files));
        assert!(
            !file.exists(),
            "the shell call's guard removes it: {file:?}"
        );

        // One scratch file per served member, all owned by the same guard: a
        // two-member command scratches two, and both go when the call ends.
        let (rewritten, files) = served_and_guard_accepted(
            "grep -n a f.txt && grep -n b g.txt",
            &ws,
            &home,
            ShellPlatform::Windows,
        );
        assert_eq!(files.len(), 2, "one spec file per served member");
        assert_eq!(
            rewritten.matches(windows::SPEC_FILE_FLAG).count(),
            2,
            "both members hand their spec over by file: {rewritten}"
        );
        drop(SpecFiles(files.clone()));
        for path in &files {
            assert!(
                !path.exists(),
                "the guard removes every spec file: {path:?}"
            );
        }
    }

    /// The serve gate follows the platform: on Windows there is no host grep to
    /// be faster than, so a single-file member is served there and stays on the
    /// real grep on unix.
    #[test]
    fn single_file_members_are_served_on_windows_only() {
        let (_tmp, ws, home) = serve_fixture();
        let command = "grep -n needle src/main.rs";

        let windows = serve_command(command, &ws, Some(&home), ShellPlatform::Windows, || true);
        assert!(
            windows.rewritten.is_some(),
            "single-file serving on Windows"
        );
        drop(SpecFiles(windows.spec_files));

        let unix = serve_command(command, &ws, Some(&home), ShellPlatform::Unix, || true);
        assert!(unix.rewritten.is_none(), "the unix perf gate stays");
        assert_eq!(unix.outcomes.len(), 1);
        assert_eq!(unix.outcomes[0].reason, "single file");
    }

    /// The shapes a read-only Windows role actually issues must clear the guard
    /// too: the engine verb is a path-qualified, quoted, extension-qualified
    /// word there, and the rewrite keeps the agent's own `cd`, glob and
    /// redirect text — a `cd <dir> && grep …` search (the dominant live shape)
    /// is served like the plain one.
    #[test]
    fn the_shapes_a_read_only_windows_role_issues_are_served() {
        let (_tmp, ws, home) = serve_fixture();
        for command in [
            "grep -rn needle .",
            "cd src && grep -n needle main.rs",
            "grep -rn --include=*.rs needle .",
            "grep -rn needle *",
            "grep \".\" src/main.rs",
        ] {
            let (rewritten, files) =
                served_and_guard_accepted(command, &ws, &home, ShellPlatform::Windows);
            assert!(rewritten.contains(ENGINE_VERB), "{command}: {rewritten}");
            drop(SpecFiles(files));
        }
    }

    /// cmd.exe's cwd family is tracked the way `cd` is — `pushd <dir>` and
    /// `chdir <dir>` both change to that directory — so a search after one is
    /// served against the right cwd. Tracking only `cd` would leave the cwd
    /// stale and the engine's own cwd gate would refuse the serve at runtime.
    /// The verb is read through the same folded key as every other list, so the
    /// interpreter's own spellings (`CD`, and a quoted verb, which cmd.exe
    /// strips) are that same builtin.
    #[test]
    fn windows_tracks_pushd_and_chdir_like_cd() {
        let (_tmp, ws, home) = serve_fixture();
        for command in [
            "pushd src && grep -rn needle .",
            "chdir src && grep -rn needle .",
            "CD src && grep -rn needle .",
            r#""cd" src && grep -rn needle ."#,
        ] {
            let (rewritten, files) =
                served_and_guard_accepted(command, &ws, &home, ShellPlatform::Windows);
            assert!(rewritten.contains(ENGINE_VERB), "{command}: {rewritten}");
            let cwd = spec_cwd(&files);
            assert!(
                cwd.ends_with("src"),
                "{command}: tracked cwd must be inside src: {cwd}"
            );
            drop(SpecFiles(files));
        }
    }

    /// A quoted `cd` target keeps its inner spaces: the member is read with
    /// cmd.exe's own tokenizer, so `cd "my dir"` tracks that directory instead
    /// of being refused as untrackable — which on this platform would refuse a
    /// servable search outright, for any workspace whose path has a space in it.
    #[test]
    fn windows_tracks_a_quoted_cd_target() {
        let (_tmp, ws, home) = serve_fixture();
        fs::create_dir_all(ws.join("my dir")).expect("my dir");
        let (rewritten, files) = served_and_guard_accepted(
            "cd \"my dir\" && grep -rn needle .",
            &ws,
            &home,
            ShellPlatform::Windows,
        );
        assert!(rewritten.contains(ENGINE_VERB), "{rewritten}");
        let cwd = spec_cwd(&files);
        assert!(
            cwd.ends_with("my dir"),
            "tracked cwd must be inside `my dir`: {cwd}"
        );
        drop(SpecFiles(files));
    }

    /// The segmenter's newline connector is re-emitted as `&`: the rewrite rides
    /// a single `/C "…"` argument, where a literal newline is the one spelling
    /// this workspace cannot measure (see the `windows` module header).
    #[test]
    fn a_newline_connector_joins_the_members_with_an_ampersand() {
        let (_tmp, ws, home) = serve_fixture();
        let (rewritten, files) = served_and_guard_accepted(
            "cd src\ngrep -rn needle .",
            &ws,
            &home,
            ShellPlatform::Windows,
        );
        assert!(
            !rewritten.contains('\n'),
            "newline in the rewrite: {rewritten}"
        );
        assert!(
            rewritten.contains(" & "),
            "members joined with `&`: {rewritten}"
        );
        let cwd = spec_cwd(&files);
        assert!(
            cwd.ends_with("src"),
            "tracked cwd must be inside src: {cwd}"
        );
        drop(SpecFiles(files));
    }

    /// The executable spelling a real Windows install has — a space and
    /// parentheses — must clear the guard too; the pins above only ever feed
    /// this binary's POSIX-shaped path.
    #[test]
    fn the_guard_accepts_a_windows_installed_executable_spelling() {
        let (_tmp, ws, _home) = serve_fixture();
        let mut files = Vec::new();
        let fragment = render_served(
            &render_only_json(),
            &[],
            ShellPlatform::Windows,
            r"C:\Program Files (x86)\MahBot\mahbot.exe",
            &mut files,
        )
        .expect("the install spelling is a quotable argument");
        let ctx = CheckContext::for_platform(&ws, ShellPlatform::Windows);
        let verdict = check_command(&fragment, &ctx);
        drop(SpecFiles(files));
        assert!(
            verdict.is_ok(),
            "read-only guard rejected the installed-executable rewrite {fragment}: {verdict:?}"
        );
    }
}

// ── Refusal pins (both platforms, no feature gate) ───────────────────────
// On Windows a command carrying a grep-family invocation is either served or
// refused; on unix an unserved member keeps falling back to the real grep, so
// nothing is refused there. Both verdicts are drivable from this host: the
// platform and the engine probe are explicit arguments (see [`serve_command`]).

#[cfg(test)]
mod refusal_pins {
    use super::super::SpecFiles;
    use super::*;

    /// Serve `command` on Windows (probe bypassed) and require a refusal whose
    /// message names its cause and says the search did not run. Returns the
    /// cause.
    fn refused(command: &str, ws: &Path, home: &Path) -> String {
        let mut serve = serve_command(command, ws, Some(home), ShellPlatform::Windows, || true);
        assert!(
            serve.rewritten.is_none(),
            "{command}: expected no rewrite (got {:?})",
            serve.rewritten
        );
        let cause = serve
            .refusal
            .take()
            .unwrap_or_else(|| panic!("{command}: expected a refusal"));
        let message = unserved_failure(&cause);
        assert!(!cause.is_empty(), "{command}: a cause is named");
        assert!(
            message.contains("did NOT run"),
            "{command}: the message must say the search did not run: {message}"
        );
        assert!(
            message.contains(&cause),
            "{command}: the message must name the cause {cause:?}: {message}"
        );
        assert!(
            serve.spec_files.is_empty(),
            "{command}: a refused command scratches no spec file"
        );
        cause
    }

    /// The probe is asked once a complete analysis has left something to serve,
    /// and a probe that answers "no" demotes exactly those members: a command
    /// carrying no search never spawns it (the reason it is a closure rather
    /// than a result), and an unavailable engine leaves the analysed member
    /// unserved with that cause instead of producing a rewrite.
    #[test]
    fn the_engine_probe_is_asked_only_when_there_is_something_to_serve() {
        let (_tmp, ws, home) = serve_fixture();
        let asked = std::cell::Cell::new(0usize);
        let probe = || {
            asked.set(asked.get() + 1);
            true
        };

        let no_search = serve_command("echo hi", &ws, Some(&home), ShellPlatform::Unix, probe);
        assert!(no_search.rewritten.is_none());
        assert_eq!(asked.get(), 0, "a command with no search must not probe");

        let servable = serve_command("grep -rn x .", &ws, Some(&home), ShellPlatform::Unix, probe);
        assert_eq!(asked.get(), 1, "a servable member asks exactly once");
        assert!(servable.rewritten.is_some());

        let unavailable = serve_command(
            "grep -rn x .",
            &ws,
            Some(&home),
            ShellPlatform::Unix,
            || false,
        );
        assert!(unavailable.rewritten.is_none());
        assert_eq!(unavailable.outcomes.len(), 1, "the member is analysed");
        assert!(!unavailable.outcomes[0].served);
        assert_eq!(unavailable.outcomes[0].reason, "engine unavailable");
    }

    #[test]
    fn windows_refuses_every_version_of_an_unserved_search() {
        let (_tmp, ws, home) = serve_fixture();
        let rows: &[(&str, &str)] = &[
            // A member in grep-verb position the serve decision rejected.
            ("grep -P x f.txt", "unsupported flag -P"),
            // A `;` outside quotes: cmd.exe reads it as an argument delimiter,
            // so the cmd.exe model refuses the line before any member exists.
            ("echo a; grep x f.txt", "unquoted `;`"),
            // A caret outside quotes: unmodellable, and a search is present.
            ("grep ^fn src/main.rs", "the whole line"),
            // A nested/compound member the engine never claims.
            ("xargs grep x f.txt", "nested grep"),
            // A `cd` the tracked-cwd model cannot resolve (`/x` is switch-shaped
            // to cmd.exe; a `-x` target would be an ordinary directory name
            // there, served statically and diverging only at runtime, where the
            // engine's own sentinel refuses).
            ("cd /x && grep x f.txt", "cd untrackable"),
            // The family's untrackable halves: `popd` returns to a directory
            // only the pushed stack knows, and a bare `pushd` is not `cd`'s
            // no-op. Neither is modelled, so the cwd after them is unknown.
            ("popd && grep -rn x .", "cd untrackable"),
            ("pushd && grep -rn x .", "cd untrackable"),
            // A command group, which cmd.exe parses itself (and which the
            // group tracking below never sees: the line is unreadable first).
            ("(grep x f.txt)", "the whole line"),
            // One servable member plus one the engine cannot serve: the whole
            // command is refused rather than half-served.
            ("grep -rn needle . && xargs grep x f.txt", "nested grep"),
            // A second grep in a pipeline consumes the engine's stream through
            // a program this platform does not have.
            ("grep -rn needle . | grep -v skip", "nested grep"),
            // A grep inside another interpreter's command line: that
            // interpreter's own reading of the payload, which the engine can
            // neither serve nor hand back to a real `grep` on this platform.
            // Every spelling of a verb is one verb (case and `.exe` folded):
            // the switch's spelling does not decide whether the search is seen.
            ("cmd /c grep -rn x .", "nested grep"),
            ("CMD /C GREP -rn x .", "nested grep"),
            ("cmd.exe /c grep -rn x .", "nested grep"),
            // A wrapped interpreter receives its payload as ONE quoted word:
            // the quoted content is the command line it will read, so a search
            // there is the same nested search however the quotes are placed.
            (r#"cmd /c "grep -rn x .""#, "nested grep"),
            ("powershell -c \"git log | grep x\"", "nested grep"),
            // A redirect glued to a word: cmd.exe splits `x>out.txt` into the
            // argument `x` plus a redirect, while the shared token classifier
            // reads an operator only where it OPENS the word — so serving it
            // would search for different text and drop the redirect. The
            // segmenter's `>&` reading (`a>&echo`) lands in the same class.
            ("grep -n x>out.txt f.txt", "redirect in `x>out.txt`"),
            ("grep -rn x>log .", "redirect in `x>log`"),
            ("grep a>&echo done", "redirect in `a>&echo`"),
            // `%…%` is expanded by cmd.exe before any program sees its argv —
            // an undefined name expands to nothing — so a pattern or a filter
            // spelling a pair cannot be served literally either. The reading is
            // per word and counts `%` characters, so `100%%` is refused too.
            ("grep -n %TEMP% f.txt", "two `%` in %TEMP%"),
            ("grep -n x --include=%x% f.txt", "two `%` in --include=%x%"),
            ("grep -n 100%% f.txt", "two `%` in 100%%"),
        ];
        for (command, cause) in rows {
            let reason = refused(command, &ws, &home);
            assert!(
                reason.contains(cause),
                "{command}: reason {reason:?} lacks {cause:?}"
            );
        }
    }

    /// The narrowing: a lone `%` and every `!` are ordinary characters to the
    /// interpreter this process spawns (`cmd /C`, no `/V:ON`), so the searches
    /// they appear in are served rather than refused.
    #[test]
    fn windows_serves_the_lone_percent_and_bang_searches() {
        let (_tmp, ws, home) = serve_fixture();
        for command in [
            r#"grep -rn "!=" src/main.rs"#,
            r#"grep -n "50%" src/main.rs"#,
            r#"grep -rn "!important" ."#,
        ] {
            let (_rewritten, files) = served(command, &ws, &home, ShellPlatform::Windows);
            drop(SpecFiles(files));
        }
    }

    /// The glued form is the refused one: a redirect word of its own is kept
    /// verbatim (cmd.exe parses it, and the shape is the interpreter's own), and
    /// a `>` inside quotes is ordinary text to the program that receives the
    /// word — the served member's text rides the spec file, never this command
    /// line.
    #[test]
    fn windows_serves_the_readable_redirect_shapes() {
        let (_tmp, ws, home) = serve_fixture();
        for command in [
            "grep -n needle f.txt > out.txt",
            "grep -n needle f.txt 2>&1",
            r#"grep -n "a>b" f.txt"#,
        ] {
            let (rewritten, files) = served(command, &ws, &home, ShellPlatform::Windows);
            assert!(rewritten.contains(ENGINE_VERB), "{command}: {rewritten}");
            drop(SpecFiles(files));
        }
    }

    /// The tolerant command-position scan reads command separators and a
    /// leading group opener, so a grep word in ARGUMENT position (`echo (grep
    /// is a tool)`) is not the search the class refuses.
    #[test]
    fn windows_does_not_refuse_a_grep_word_in_argument_position() {
        let (_tmp, ws, home) = serve_fixture();
        for command in ["echo hi & echo (grep is a tool)", "echo (grep x f.txt)"] {
            let serve = serve_command(command, &ws, Some(&home), ShellPlatform::Windows, || true);
            assert!(
                serve.refusal.is_none(),
                "{command}: must not be refused: {:?}",
                serve.refusal
            );
            assert!(serve.rewritten.is_none(), "{command}: nothing to serve");
        }
    }

    /// The double-quoted spelling of the same search is readable (the caret is
    /// inside quotes), so it is served, not refused.
    #[test]
    fn windows_serves_the_quoted_caret_spelling() {
        let (_tmp, ws, home) = serve_fixture();
        let mut serve = serve_command(
            "grep \"^fn \" src/main.rs",
            &ws,
            Some(&home),
            ShellPlatform::Windows,
            || true,
        );
        assert!(serve.refusal.is_none(), "{:?}", serve.refusal);
        assert!(serve.rewritten.is_some(), "a serveable search");
        drop(SpecFiles(std::mem::take(&mut serve.spec_files)));
    }

    /// The Windows exemption list of the module header: the shapes left to the
    /// platform as written — neither served nor refused, so the platform's own
    /// program is what runs.
    #[test]
    fn windows_leaves_every_exempt_shape_alone() {
        let (_tmp, ws, home) = serve_fixture();
        for command in [
            "echo hi",
            "git commit -m \"grep fix\"",
            "echo grep",
            // A program spelled with a path is a file, not this platform's
            // `grep` verb, so the segment runs as written — the reading unix
            // gives `./grep` (the engine cannot tell whether such a program
            // exists, and the interpreter's own error names it if it does not).
            r".\grep -n x f.txt",
            r"C:\tools\grep.exe -rn x .",
            // The search-owning list is folded like every other verb list, so
            // the exemption holds for any spelling of the program.
            "GIT GREP x",
            // A grep word glued to a leading `{`: `{grep` names no command on
            // either platform, so the line carries no invocation to claim.
            "{grep x f.txt",
            // A redirect glued to the verb extends the command word (`grep>x`),
            // which names no search either.
            "grep>x f.txt",
            // An unlisted program's argument list is data to this reading.
            "foo grep x",
            // The `\;` spelling stops the cmd reading before any member exists;
            // the `+` spelling of the same `find` refuses instead.
            r"find . -exec grep x {} \;",
        ] {
            let serve = serve_command(command, &ws, Some(&home), ShellPlatform::Windows, || true);
            assert!(
                serve.refusal.is_none(),
                "{command}: must not be refused: {:?}",
                serve.refusal
            );
            assert!(serve.rewritten.is_none(), "{command}: nothing to serve");
        }

        // `git grep x` is git's own search: recorded as a skipped member, and
        // deliberately left unserved-but-allowed rather than refused.
        let serve = serve_command(
            "git grep x",
            &ws,
            Some(&home),
            ShellPlatform::Windows,
            || true,
        );
        assert!(serve.refusal.is_none(), "{:?}", serve.refusal);
        assert!(serve.rewritten.is_none());
        assert_eq!(serve.outcomes.len(), 1, "recorded as a skipped member");
        assert!(!serve.outcomes[0].served);

        // `ssh host grep …` searches another machine: again no member to serve
        // and, crucially, nothing to refuse — this platform can run the command
        // as written.
        let serve = serve_command(
            "ssh host grep -rn x /srv",
            &ws,
            Some(&home),
            ShellPlatform::Windows,
            || true,
        );
        assert!(serve.refusal.is_none(), "{:?}", serve.refusal);
        assert!(serve.rewritten.is_none());
        assert_eq!(serve.outcomes.len(), 1, "recorded as a skipped member");
    }

    /// Unix never refuses and never produces a rewrite for these rows: the
    /// member the engine cannot serve keeps running the real `grep`.
    #[test]
    fn unix_keeps_the_fallback_and_never_refuses() {
        let (_tmp, ws, home) = serve_fixture();
        let rows = [
            "grep -P x f.txt",
            "echo a; grep x f.txt",
            "grep ^fn src/main.rs",
            "xargs grep x f.txt",
            "cd /x && grep x f.txt",
            "(grep x f.txt)",
            "git grep x",
        ];
        for command in rows {
            let serve = serve_command(command, &ws, Some(&home), ShellPlatform::Unix, || true);
            assert!(
                serve.refusal.is_none(),
                "{command}: unix keeps the fallback: {:?}",
                serve.refusal
            );
            assert!(
                serve.rewritten.is_none(),
                "{command}: nothing to serve (got {:?})",
                serve.rewritten
            );
        }
    }

    /// The routing hands the analysis no home on Windows — nothing there
    /// expands `~`, so the `home` arms of `resolve_operand`/`resolve_cd` are
    /// unreachable — and the placeholder it substitutes for the one the analysis
    /// takes must therefore never be consulted: a member whose cwd tracking and
    /// operands resolve statically is served with no home at all. Unix keeps its
    /// documented reading of the same absence (nothing that could expand `~` is
    /// served, and nothing is refused either).
    #[test]
    fn windows_serves_without_a_home() {
        let (_tmp, ws, _home) = serve_fixture();
        let mut windows = serve_command(
            "cd src && grep -n needle main.rs",
            &ws,
            None,
            ShellPlatform::Windows,
            || true,
        );
        assert!(windows.refusal.is_none(), "{:?}", windows.refusal);
        let rewritten = windows.rewritten.take().expect("served with no home");
        assert!(rewritten.contains(ENGINE_VERB), "{rewritten}");
        let cwd = spec_cwd(&windows.spec_files);
        assert!(cwd.ends_with("src"), "tracked cwd: {cwd}");
        drop(SpecFiles(windows.spec_files));

        let unix = serve_command("grep -rn needle .", &ws, None, ShellPlatform::Unix, || true);
        assert!(unix.rewritten.is_none(), "unix demotes without a home");
        assert!(unix.refusal.is_none(), "and refuses nothing there");
    }

    /// The interpreter names `cmd`, `powershell` and `pwsh` are list entries on
    /// unix too: a `cmd /c grep …` line records a skipped member there and still
    /// refuses nothing, since nothing about its serve decision or its text
    /// changes.
    #[test]
    fn unix_records_a_wrapped_interpreter_as_a_skip_without_refusing() {
        let (_tmp, ws, home) = serve_fixture();
        let serve = serve_command(
            "cmd /c grep x f.txt",
            &ws,
            Some(&home),
            ShellPlatform::Unix,
            || true,
        );
        assert!(serve.refusal.is_none(), "{:?}", serve.refusal);
        assert!(serve.rewritten.is_none());
        assert_eq!(serve.outcomes.len(), 1);
        assert!(!serve.outcomes[0].served);
        assert_eq!(serve.outcomes[0].reason, "nested grep");
    }

    /// The unix twin of the mixed row above: the servable member is still
    /// served while the rejected sibling runs the real `grep`.
    #[test]
    fn unix_still_serves_a_sibling_of_a_rejected_member() {
        let (_tmp, ws, home) = serve_fixture();
        let mut serve = serve_command(
            "grep -rn needle . && xargs grep x f.txt",
            &ws,
            Some(&home),
            ShellPlatform::Unix,
            || true,
        );
        assert!(serve.refusal.is_none(), "{:?}", serve.refusal);
        let rewritten = serve.rewritten.take().expect("the sibling is served");
        assert!(rewritten.contains(ENGINE_VERB), "{rewritten}");
        assert!(rewritten.contains("xargs grep x f.txt"), "{rewritten}");
    }

    /// The `%…%` reading is the cmd.exe one alone: on unix a `%` and a `!`
    /// are ordinary characters, so every one of these is served — including the
    /// rows Windows still refuses for their `%` pair.
    #[test]
    fn unix_serves_the_percent_and_bang_searches() {
        let (_tmp, ws, home) = serve_fixture();
        for command in [
            "grep -rn %TEMP% .",
            "grep -rn x --include=%x% .",
            "grep -rn \"a!b\" .",
        ] {
            let mut serve = serve_command(command, &ws, Some(&home), ShellPlatform::Unix, || true);
            assert!(serve.refusal.is_none(), "{command}: {:?}", serve.refusal);
            assert!(
                serve.rewritten.is_some(),
                "{command}: served on unix (got {:?})",
                serve.rewritten
            );
            drop(SpecFiles(std::mem::take(&mut serve.spec_files)));
        }
    }
}

// ── The unix word/operator reading (unix-gated: parsing plus a real `sh` run) ─
// The reading the module header states, observed from both sides: which text
// the member is served on, and that the operator is handed back to the shell
// to perform. The wider differential rows live in the parity matrix and the
// e2e bench.

#[cfg(all(test, unix))]
mod unix_operator_pins {
    use super::test_support::engine_run;
    use super::*;
    use std::process::Command;

    /// What the served member was read as, in one labelled line: the text it
    /// searches, the raw redirect spellings returned to the shell for it, and
    /// its operands. A row is one spelling, so the family reads as a table and
    /// a mismatch names the field it came from.
    fn reading(command: &str, ws: &Path, home: &Path) -> String {
        let analyzed = analyze_command(command, ws, home, ShellPlatform::Unix, false)
            .unwrap_or_else(|e| panic!("{command}: expected servable, got {}", e.reason));
        assert_eq!(analyzed.specs.len(), 1, "{command}: one served member");
        let OutSegment::Served { redirects, .. } = &analyzed.segments[0].0 else {
            panic!("{command}: expected a served member");
        };
        let operands: Vec<&str> = analyzed.specs[0]
            .operands
            .iter()
            .map(|o| o.display.as_str())
            .collect();
        format!(
            "search={} redirect={} operands={}",
            analyzed.specs[0].patterns.join(" "),
            redirects.join(" "),
            operands.join(" ")
        )
    }

    /// `sh`'s comment, read as the comment it is: an unquoted `#` opening a word
    /// comments out the rest of its line, so the search before it is served on
    /// exactly the words the shell would have passed and everything inside the
    /// comment — a backgrounding `&` included — goes with it, while the lines
    /// below still run.
    #[test]
    fn the_comment_takes_the_rest_of_its_line_out_of_the_command() {
        let (tmp, ws, home) = serve_fixture();
        // (command, what the member was read as): the comment is neither a word
        // of the search nor an operator to hand back.
        let rows: &[(&str, &str)] = &[
            // The `&` and the echo after the `#` are comment text: what is left
            // to serve is the search alone.
            (
                "grep -rn needle . # note & echo TAIL-RAN",
                "search=needle redirect= operands=.",
            ),
            // An unbalanced quote inside a comment is inert, as it is in `sh`.
            (
                "grep -rn needle . # don't",
                "search=needle redirect= operands=.",
            ),
            // A `#` that opens no word stays ordinary text: the mid-word, quoted
            // and escaped spellings are all the same word to `sh`.
            ("grep -rn a#b .", "search=a#b redirect= operands=."),
            ("grep -rn 'a#b' .", "search=a#b redirect= operands=."),
            (r"grep -rn a\#b .", "search=a#b redirect= operands=."),
        ];
        for (command, expected) in rows {
            assert_eq!(reading(command, &ws, &home), *expected, "{command}");
        }

        // The other side of the reading, under a real `sh` with a stand-in for
        // the engine: nothing the comment swallowed is ever run, the `&` inside
        // it never backgrounds the member — the call reports the member's own
        // status (`ENGINE_EXIT`) — and the line below the comment still runs.
        // The middle row is the same line without the comment, where the tail
        // really does run.
        let engine = engine_stub(tmp.path());
        // (command, stdout, status, whether the swallowed tail survives)
        let rows: &[(&str, &str, i32, bool)] = &[
            (
                "grep -rn needle . # note & echo TAIL-RAN",
                "ENGINE-OUT\n",
                ENGINE_EXIT,
                false,
            ),
            (
                "grep -rn needle . & wait; echo TAIL-RAN",
                "ENGINE-OUT\nTAIL-RAN\n",
                0,
                true,
            ),
            (
                "grep -rn needle . # note & echo TAIL-RAN\nls f.txt",
                "ENGINE-OUT\nf.txt\n",
                0,
                false,
            ),
        ];
        for (command, stdout, status, tail_kept) in rows {
            let analyzed = analyze_command(command, &ws, &home, ShellPlatform::Unix, false)
                .unwrap_or_else(|e| panic!("{command}: expected servable, got {}", e.reason));
            let jsons: Vec<String> = analyzed.specs.iter().map(spec_json).collect();
            let (rewritten, _) =
                join_rewritten(&analyzed.segments, &jsons, ShellPlatform::Unix, &engine)
                    .expect("the unix join never refuses");
            assert_eq!(
                rewritten.contains("TAIL-RAN"),
                *tail_kept,
                "{command}: the comment takes the tail with it ({rewritten})"
            );
            let out = Command::new("/bin/sh")
                .arg("-c")
                .arg(&rewritten)
                .current_dir(&ws)
                .output()
                .expect("sh runs");
            assert_eq!(
                String::from_utf8_lossy(&out.stdout),
                *stdout,
                "{command}: stdout (rewrite: {rewritten})"
            );
            assert_eq!(
                out.status.code(),
                Some(*status),
                "{command}: status (rewrite: {rewritten})"
            );
        }
    }

    /// The whole glued-operator family, read the way `sh` reads it: the operator
    /// ends the word before it (or takes that word's digits as an fd prefix) and
    /// is handed back to the shell verbatim, while quoted and escaped operators
    /// stay ordinary text.
    #[test]
    fn the_glued_operator_family_is_read_the_way_sh_reads_it() {
        let (_tmp, ws, home) = serve_fixture();
        let rows: &[(&str, &str)] = &[
            // The separated spellings the glued rows must land on.
            (
                "grep -rn needle > out.txt .",
                "search=needle redirect=> out.txt operands=.",
            ),
            (
                "grep -rn needle 2> out.txt .",
                "search=needle redirect=2> out.txt operands=.",
            ),
            (
                "grep -rn needle>out.txt .",
                "search=needle redirect=>out.txt operands=.",
            ),
            (
                "grep -rn needle>>out.txt .",
                "search=needle redirect=>>out.txt operands=.",
            ),
            (
                "grep -rn needle<f.txt .",
                "search=needle redirect=<f.txt operands=.",
            ),
            (
                "grep -rn needle>&1 .",
                "search=needle redirect=>&1 operands=.",
            ),
            // `<&` is the input-side fd dup: one operator too.
            (
                "grep -rn needle . <&2",
                "search=needle redirect=<&2 operands=.",
            ),
            (
                "grep -rn 2<&1 needle .",
                "search=needle redirect=2<&1 operands=.",
            ),
            // `&>` is one operator, and the word before it stays part of the
            // search (unlike an fd prefix).
            (
                "grep -rn needle&>out.txt .",
                "search=needle redirect=&>out.txt operands=.",
            ),
            (
                "grep -rn needle&>>out.txt .",
                "search=needle redirect=&>>out.txt operands=.",
            ),
            // The digits belong to the operator only when everything before it
            // is digits: `2>log` redirects fd 2, `needle2>log` is the word
            // `needle2` plus a redirect to `log`.
            (
                "grep -rn 2>out.txt needle .",
                "search=needle redirect=2>out.txt operands=.",
            ),
            (
                "grep -rn 10>out.txt needle .",
                "search=needle redirect=10>out.txt operands=.",
            ),
            (
                "grep -rn needle2>out.txt .",
                "search=needle2 redirect=>out.txt operands=.",
            ),
            // Quoted and escaped operators are ordinary text.
            ("grep -rn 'a>b' .", "search=a>b redirect= operands=."),
            ("grep -rn \"x>log\" .", "search=x>log redirect= operands=."),
            (r"grep -rn a\>b .", "search=a>b redirect= operands=."),
        ];
        for (command, expected) in rows {
            assert_eq!(reading(command, &ws, &home), *expected, "{command}");
        }
    }

    /// The member searches exactly the text the same search spelled with the
    /// operator separate searches — the words `sh` would have passed to `grep`,
    /// with the operator gone from the argv.
    #[test]
    fn the_engine_searches_the_text_the_separated_spelling_searches() {
        let (_tmp, ws, home) = serve_fixture();
        let expected = "f.txt:needle\nsrc/main.rs:fn needle() {}\n";
        for command in [
            "grep needle f.txt src/main.rs",
            "grep needle>out.txt f.txt src/main.rs",
            "grep needle>>out.txt f.txt src/main.rs",
            "grep needle<f.txt f.txt src/main.rs",
            "grep needle&>out.txt f.txt src/main.rs",
            "grep 2>out.txt needle f.txt src/main.rs",
        ] {
            let analyzed = analyze_command(command, &ws, &home, ShellPlatform::Unix, false)
                .unwrap_or_else(|e| panic!("{command}: expected servable, got {}", e.reason));
            let (out, err, code) = engine_run(&analyzed.specs[0]);
            assert_eq!(code, 0, "{command}: exit");
            assert_eq!(
                String::from_utf8_lossy(&out),
                expected,
                "{command}: searched text"
            );
            assert!(err.is_empty(), "{command}: stderr: {err:?}");
        }
    }

    const ENGINE_OUT: &str = "ENGINE-OUT";
    const ENGINE_ERR: &str = "ENGINE-ERR";
    /// The stand-in's own exit status: a row can tell whether the shell waited
    /// for the member or left it running.
    const ENGINE_EXIT: i32 = 7;

    /// A stand-in for the engine binary the rewrite names — the renderer takes
    /// that path as an argument — answering on both streams so a row can see
    /// which stream the shell routed where.
    fn engine_stub(dir: &Path) -> String {
        let path = dir.join("mahbot-engine-stub");
        fs::write(
            &path,
            format!(
                "#!/bin/sh\nprintf '%s\\n' {ENGINE_OUT}\nprintf '%s\\n' {ENGINE_ERR} >&2\nexit {ENGINE_EXIT}\n"
            ),
        )
        .expect("stub written");
        let mut perms = fs::metadata(&path).expect("stub").permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o755);
        fs::set_permissions(&path, perms).expect("stub executable");
        path.to_string_lossy().into_owned()
    }

    /// The other side of the reading, observed by running the rendered rewrite
    /// under a real `sh`: the shell performs the operator the command spelled —
    /// the member's stream lands in the file it named — and a backgrounded
    /// member leaves the commands after its `&` running.
    #[test]
    fn the_shell_performs_the_write_and_keeps_the_backgrounded_tail() {
        let (tmp, ws, home) = serve_fixture();
        let engine = engine_stub(tmp.path());
        // (command, target, what the target receives, stdout)
        let rows: &[(&str, &str, &str, &str)] = &[
            ("grep -rn needle>out.txt .", "out.txt", "ENGINE-OUT\n", ""),
            ("grep -rn needle>>out.txt .", "out.txt", "ENGINE-OUT\n", ""),
            (
                "grep -rn 2>err.txt needle .",
                "err.txt",
                "ENGINE-ERR\n",
                "ENGINE-OUT\n",
            ),
            (
                "grep -rn needle>out.txt . & wait; echo AFTER",
                "out.txt",
                "ENGINE-OUT\n",
                "AFTER\n",
            ),
        ];
        for (command, target, in_target, stdout) in rows {
            let _ = fs::remove_file(ws.join(target));
            let analyzed = analyze_command(command, &ws, &home, ShellPlatform::Unix, false)
                .unwrap_or_else(|e| panic!("{command}: expected servable, got {}", e.reason));
            let jsons: Vec<String> = analyzed.specs.iter().map(spec_json).collect();
            let (rewritten, _) =
                join_rewritten(&analyzed.segments, &jsons, ShellPlatform::Unix, &engine)
                    .expect("the unix join never refuses");
            let out = Command::new("/bin/sh")
                .arg("-c")
                .arg(&rewritten)
                .current_dir(&ws)
                .output()
                .expect("sh runs");
            assert_eq!(
                fs::read_to_string(ws.join(target)).unwrap_or_default(),
                *in_target,
                "{command}: `sh` writes the member's stream into {target} (rewrite: {rewritten})"
            );
            assert_eq!(
                String::from_utf8_lossy(&out.stdout),
                *stdout,
                "{command}: stdout (rewrite: {rewritten})"
            );
        }
    }

    /// A member backgrounded as the last thing in its line: the `&` itself
    /// reaches the shell, so the call returns on the shell's own status rather
    /// than the member's — the stand-in exits 7, which a foregrounded member
    /// would report.
    #[test]
    fn a_trailing_ampersand_backgrounds_the_served_member() {
        let (tmp, ws, home) = serve_fixture();
        let engine = engine_stub(tmp.path());
        let command = "grep -rn needle . &";
        let analyzed = analyze_command(command, &ws, &home, ShellPlatform::Unix, false)
            .unwrap_or_else(|e| panic!("{command}: expected servable, got {}", e.reason));
        let jsons: Vec<String> = analyzed.specs.iter().map(spec_json).collect();
        let (rewritten, _) =
            join_rewritten(&analyzed.segments, &jsons, ShellPlatform::Unix, &engine)
                .expect("the unix join never refuses");
        assert!(
            rewritten.ends_with(" &"),
            "the `&` rides the rewrite: {rewritten}"
        );
        let out = Command::new("/bin/sh")
            .arg("-c")
            .arg(&rewritten)
            .current_dir(&ws)
            .output()
            .expect("sh runs");
        assert_eq!(
            out.status.code(),
            Some(0),
            "the shell's status, not the backgrounded member's: {rewritten}"
        );
    }

    /// `>(…)`/`<(…)` is process substitution — a word the shell builds from a
    /// command — never a redirection: the member stays on the real `grep`
    /// rather than being served on a word nothing can resolve.
    #[test]
    fn process_substitution_is_never_a_redirect() {
        let (_tmp, ws, home) = serve_fixture();
        for command in [
            "grep -rn needle <(cat f.txt) .",
            "grep -rn needle . >(cat f.txt)",
        ] {
            let serve = serve_command(command, &ws, Some(&home), ShellPlatform::Unix, || true);
            assert!(serve.refusal.is_none(), "{command}: unix never refuses");
            assert!(
                serve.rewritten.is_none(),
                "{command}: nothing to serve (got {:?})",
                serve.rewritten
            );
            assert_eq!(serve.outcomes.len(), 1, "{command}: one analyzed member");
            assert_eq!(
                serve.outcomes[0].reason, "process substitution",
                "{command}"
            );
        }
    }
}
