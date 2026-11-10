//! Transparent grep/egrep/fgrep interception for the shell tool in both read-only
//! and full modes (the engine is inherently read-only). The read-only branch
//! re-validates the rewrite through `shell::readonly`, whose guard must accept the
//! quoted absolute executable path the rewrite names.
//!
//! Analysis is per platform and reads the value the spawn side reads
//! ([`SHELL_PLATFORM`]): `sh -c` on unix, `cmd.exe /C` on Windows, whose
//! segmenting, quoting, verb dispatch and `cd` grammar `windows` owns. Platform
//! decisions take that value as a parameter (a host test lane drives either
//! platform); verb lists are consulted through the platform's key
//! ([`list_key`]/[`windows::verb_key`]) so the interpreter's case- and
//! `.exe`-insensitive dispatch applies; only OS APIs (`exec`, `creation_flags`,
//! the SIGPIPE disposition) are `#[cfg]`-gated.
//!
//! Servable invocations are rewritten to a hidden `__grep-engine` subcommand of
//! this binary (dispatched before instance-lock acquisition in `main()`), which
//! runs the ripgrep substrate (grep-regex/grep-searcher + ignore-crate rg-default
//! walk exclusions). Substitution is per-segment: an unservable grep (single-file
//! perf gate, unsupported flag, compound/nested shape, …) stays verbatim while a
//! servable sibling is served; anything not provably safe is left unrewritten,
//! running as this platform reads it ([`super::windows_line`]).
//!
//! On Windows that tolerance is gone: an unserved member would run a `grep` the
//! platform lacks, and "not recognized" is the very status "no match" produces, so
//! a command carrying a grep-family invocation is either fully served or refused
//! ([`unserved_failure`], [`GrepServe::refusal`]). Refusal is fail-closed on words, not
//! meanings: a grep-family word in a followed program's argument list refuses
//! ([`GREP_INTRODUCERS`]) unless that program owns the search
//! ([`SEARCH_OWNING_VERBS`]), and an unreadable line refuses when a command-position
//! search is still visible ([`unreadable_line_search`]); both over-refusals are
//! deliberate and the loud failure names the cause. The shapes left to the platform
//! carry no grep member in command position: a search another program owns
//! ([`SEARCH_OWNING_VERBS`] — `git grep`, `docker … grep`, `ssh host grep`), one in the
//! argument list of a program no list names (`foo grep x`), a command word that is no
//! grep verb (a path-spelled program, `{grep`, a redirect glued to the verb), and the
//! `find … -exec grep … \;` spelling, whose unquoted `;` stops the cmd reading.
//!
//! There the rewrite is not handed to `cmd.exe` as one line: a member calling this
//! service's own windowed image becomes the runner's own step ([`plan`]), while a
//! member the runner cannot reproduce is refused. The per-member decisions are
//! rendered twice — as the line and as that plan — and a disagreement would report
//! output from somewhere the agent did not look; `plan_pins` holds them together.
//!
//! The engine re-validates and hands the member back to the real grep on any
//! runtime doubt: in place via `exec` on unix; on Windows it reports the reason on
//! stderr and exits with the sentinel code the parent refuses on, alongside
//! [`ENGINE_REFUSAL_MARKER`], which keeps the refusal recognisable when a pipeline
//! tail masks the status.
//!
//! The fast matcher (SIMD literal prefilter) and parallel recursive walk let
//! cross-file ordering differ from host BSD grep (in-file ordering is stable).
//! Approved deltas: walks skip hidden/gitignored content (rg defaults; a served
//! tail like `grep -rn … | wc -l` sees that filtered stream), `-o` + alternation
//! stays fail-closed, and a redirected walk hands an entry over in runs of
//! [`WALK_CHUNK`], so an entry with more than a chunk of matches has its records
//! split by another file's — the cross-file order the walk rows already relax, at
//! chunk instead of file grain (an entry under the chunk stays contiguous). A
//! context search's between-files `--` rides an entry's first hand-off: its count
//! (one per printing entry, minus the first) stays exact, its placement does not.
//! Not buffering whole files is what buys the grain: a serial walk measured
//! 2-4.5x the wall time (≈105 ms vs ≈31 ms for `grep -rn … src > file`).
//!
//! On unix the shell reads segments and words: unquoted `<`/`>` starts a
//! redirection (digits before it belong to it only when everything before is
//! digits), `&>` is one operator with the prior word kept, a bare `&` backgrounds
//! the member, an unquoted `#` opening a word drops the rest of the line, and
//! `>(…)`/`<(…)` is process substitution, never a redirect. Operators ride the
//! rewrite verbatim so the surrounding `sh` performs the write; Windows refuses
//! the glued spelling ([`windows::has_glued_redirect`]). Because the write is the
//! shell's, writer decisions follow [`StdoutDest`], the real destination.
//!
//! The matching model was written against host BSD grep under a pinned
//! `LC_ALL=C.UTF-8`, and the macOS-gated differential matrix is its authoritative
//! equivalence proof there (recursive-walk rows compare as sorted line-sets,
//! everything else byte-exact). Intercepted commands run in the owner's
//! environment, whose locale is unknown, so a served member must also clear the
//! runtime gate ([`parity`]): the fast path stands only for a shape the product
//! established to agree under the locale in effect; anything else keeps the real
//! `grep`. A member the line puts in another environment (a preceding
//! `export`/`unset`/`eval`/`source`) is not served ([`changes_the_environment`]).
//! Windows has no such gate (no system `grep`); [`windows::fnmatch`] is pinned
//! differentially against this host's `fnmatch`. Grep decisions go to the
//! `grep_telemetry` table (logs DB), not the general log stream.

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use grep_matcher::Matcher;
use serde::{Deserialize, Serialize};

use crate::tools::path::shell_quote;
use crate::tools::shell::SHELL_PIPE_READ_CAP;
use crate::tools::shell::plan::{self, Plan};
use crate::tools::shell::scan::CdScan;
use crate::tools::shell::scan::strip_heredoc_bodies;
use crate::util::UnwrapPoison;
use crate::util::is_word_char;

use super::{REFUSAL_FRAME, SHELL_PLATFORM, ShellPlatform};

/// The cmd.exe line model, read by the runner's own decomposition too ([`super::plan`]).
pub(super) mod windows;

/// The runtime locale-parity gate: the fast path may stand only where that module's
/// battery established agreement under the locale actually in effect.
mod parity;

/// The parity hooks the owner-environment reader drives (see [`parity`]).
pub(crate) use self::parity::{invalidate_verdict_for, refresh_if_environment_changed};

/// Hidden subcommand name dispatched before lock acquisition (like `debug`).
const ENGINE_VERB: &str = "__grep-engine";
/// Spec JSON protocol version; a mismatch makes the engine fall back. Bumped for
/// the `stdin`/`report_stream_bytes` fields: a spec from a swapped binary is caught
/// here (both directions) and execs the real grep via the fallback argv, before any
/// stdin read, so the exec'd grep reads the producer's pipe authentically.
const PROTOCOL_VERSION: u32 = 5;
/// NUL-detection window for binary files (FreeBSD grep reads 32 KiB).
const BINARY_WINDOW: usize = 32 * 1024;
/// Engine self-cap on written output, applied on a captured member where the shell
/// pipe reader caps at the same value (see [`StdoutDest`]).
const OUTPUT_CAP: usize = SHELL_PIPE_READ_CAP;
/// Sentinel exit code: the engine could not serve and could not exec grep either.
/// Unix re-runs the original command on it; Windows refuses the call (no system
/// `grep` to re-run) and reports the engine's own stderr reason to the agent.
pub(super) const ENGINE_FAILED_EXIT: i32 = 3;
/// Exit code for a member whose downstream pipe closed. Unix's default SIGPIPE
/// disposition produces 128+SIGPIPE and leaves this unused; Windows has no SIGPIPE,
/// so the equivalent `BrokenPipe` write error exits with the same code — a
/// `grep … | head` tail must stop the walk, not let it scan the whole tree.
#[cfg_attr(unix, expect(dead_code, reason = "the unix close path is SIGPIPE"))]
const BROKEN_PIPE_EXIT: i32 = 128 + 13;
/// Stderr signature of a stale self-update binary (one lacking the hidden
/// subcommand) running full main() and dying at instance-lock acquisition — its
/// exit 1 is a legitimate grep no-match code, so the parent treats this message
/// like the sentinel exit: unix re-runs the original command, Windows refuses.
pub(super) const STALE_BINARY_LOCK_MSG: &str = crate::self_update::INSTANCE_REFUSAL_PREFIX;
/// Engine stderr marker on a run that could not serve its member. The parent
/// refuses a Windows call on it even when a pipeline tail masked the sentinel
/// status (`grep … | tail -1`), which would otherwise look like an empty match set.
/// It is emitted on its own line and recognised by line equality, never as a
/// substring, so a matched line echoing the token is no refusal and the cause
/// extraction still reads the human reason around it. Three member-side spellings
/// lose it: merging stderr into the stream, redirecting fd 2 to a file, and a
/// preceding member flooding the parent's 256 KB stderr cap (`SHELL_PIPE_READ_CAP`)
/// — an unserved search can then surface as an empty result (accepted residual).
/// A fourth loss is outside the engine: cmd.exe failing to launch the named program
/// carries the interpreter's "not recognized" status and no marker — also accepted.
pub(super) const ENGINE_REFUSAL_MARKER: &str = "__mahbot_grep_engine_unserved__";
/// Engine stderr marker carrying the byte count consumed from a stdin-fed stream
/// (best-effort: -m/-l early stops report the consumed prefix, SIGPIPE-killed chains
/// may flush nothing, stderr merges suppress it). The shell tool strips the line and
/// logs the count — per-call stream bytes are recorded nowhere else.
const STREAM_SIZE_MARKER: &str = "__mahbot_stream_bytes__";
/// Specs larger than this are not served on unix: the payload rides a single
/// argv entry there (argv-size hygiene).
const MAX_SPEC_JSON: usize = 64 * 1024;
/// The Windows limit: the payload rides a scratch file, not the command line, so
/// only a pathological expansion hits it — the argv bound above would refuse
/// ordinary `grep -rn <pat> *` searches, large but perfectly servable.
const MAX_SPEC_FILE: usize = 4 * 1024 * 1024;
/// Searcher line-buffer cap; exceeding it is a grep-style error (exit 2).
const HEAP_LIMIT: usize = 512 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EngineSpec {
    version: u32,
    verb: String,
    mode: MatchMode,
    flags: GrepFlags,
    /// Ordered --include/--exclude filters; last match wins (BSD semantics).
    filters: Vec<(bool, String)>,
    exclude_dir: Vec<String>,
    /// Engine-dialect patterns (BRE translated, ERE/Fixed as-is).
    patterns: Vec<String>,
    operands: Vec<Operand>,
    cwd: String,
    /// Original post-expansion argv for the exec-in-place fallback (unix only).
    fallback: Vec<String>,
    /// The member feeds a pipeline tail: its stdout must not be capped, so downstream
    /// members see the full stream. The other destinations are read on the engine side
    /// from its own fd 1 (see [`StdoutDest`]), not carried here.
    #[serde(default)] // lenient: a swapped-binary spec reaches the version check
    piped: bool,
    /// Non-first pipeline member fed by a producer's stdout via stdin (no operands).
    #[serde(default)]
    stdin: bool,
    /// Emit the stream-size marker to stderr (only when stderr still reaches us).
    #[serde(default)]
    report_stream_bytes: bool,
}

/// Where the member's stdout actually goes — the one fact every writer decision
/// follows, read from this process's fd 1 plus the spec's pipeline membership.
/// Asking the destination rather than the redirect spellings covers every spelling
/// by construction (a line-wide `exec > file` included) and needs no protocol field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StdoutDest {
    Capture,
    Tail,
    File,
}

impl StdoutDest {
    /// fd 1 separates a pipe from any real sink ([`stdout_is_pipe`]); `piped` says which.
    fn of(spec: &EngineSpec, fd1_is_pipe: bool) -> Self {
        if !fd1_is_pipe {
            StdoutDest::File
        } else if spec.piped {
            StdoutDest::Tail
        } else {
            StdoutDest::Capture
        }
    }

    fn limit(self) -> Option<usize> {
        match self {
            StdoutDest::Capture => Some(OUTPUT_CAP),
            StdoutDest::Tail | StdoutDest::File => None,
        }
    }
}

/// Whether fd 1 is a pipe (the shell tool's capture, a plan member's spawn, or the
/// shell's own `|`). Every other stdout is a sink the user reads, so the engine must
/// write all of it. A redirect whose target is itself a pipe reads as a pipe here —
/// one fd cannot tell the capture pipe from a named one — so it keeps the cap a
/// captured member has, and a `fstat`/`GetFileType` that cannot answer reads as a
/// pipe too: capped is the conservative answer, never a licence to write unbounded.
#[cfg(unix)]
fn stdout_is_pipe() -> bool {
    let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `fstat` writes only through the pointer it is given, and on
    // success initialises the whole `stat`.
    if unsafe { libc::fstat(libc::STDOUT_FILENO, st.as_mut_ptr()) } != 0 {
        return true;
    }
    // SAFETY: the call above returned 0, so `st` is initialised.
    let mode = unsafe { st.assume_init() }.st_mode;
    mode & libc::S_IFMT == libc::S_IFIFO
}

#[cfg(windows)]
fn stdout_is_pipe() -> bool {
    use windows_sys::Win32::Storage::FileSystem::{FILE_TYPE_PIPE, FILE_TYPE_UNKNOWN, GetFileType};
    use windows_sys::Win32::System::Console::{GetStdHandle, STD_OUTPUT_HANDLE};
    // SAFETY: both are queries on this process's own standard handle.
    let kind = unsafe { GetFileType(GetStdHandle(STD_OUTPUT_HANDLE)) };
    // An unusable handle answers FILE_TYPE_UNKNOWN, which is no licence to write
    // unbounded: the capped behaviour is the conservative reading.
    kind == FILE_TYPE_PIPE || kind == FILE_TYPE_UNKNOWN
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Operand {
    display: String,
    resolved: String,
    trailing_slash: bool,
}

/// One analyzed grep member's serve decision (for grep telemetry).
#[derive(Debug)]
pub(super) struct GrepOutcome {
    pub served: bool,
    pub reason: String,
    pub recursive: bool,
    pub piped: bool,
    pub operand_count: usize,
    pub flags: String,
}

#[derive(Debug)]
pub(super) struct GrepServe {
    /// The rewritten command; `None` when no grep was served, which leaves the command
    /// it was handed to run — except where a Windows demotion carries a cause and the
    /// caller refuses it ([`GrepServe::refusal`]).
    pub rewritten: Option<String>,
    pub outcomes: Vec<GrepOutcome>,
    pub spec_files: Vec<PathBuf>,
    /// The rewrite decomposed into steps the runner spawns itself. `Some` only on
    /// Windows and only when a rewrite was produced: a demotion (an unserved member,
    /// an unavailable engine, a payload too large, a plan shape the runner refuses)
    /// carries no plan, and on Windows the caller refuses it when it carries a cause.
    /// See [`plan`] for the guarantee and for what is deliberately refused.
    pub plan: Option<Plan>,
    /// The short cause of the search this platform could not serve. Always `None` on
    /// unix, where an unserved member keeps falling back to the real `grep`; on Windows
    /// the caller turns it into a tool error through [`unserved_failure`].
    pub refusal: Option<String>,
}

/// The agent-facing failure for a command the engine or the runner refuses on
/// Windows (the value of [`GrepServe::refusal`] and the engine's own reported
/// failure). It becomes the tool error, so it must be unmistakable for a result:
/// it names the cause and says nothing ran rather than that the search matched
/// nothing. Every refusal this engine and its runner render comes through here.
///
/// The frame is shared with the runner's own refusal ([`plan::refusal_message`]):
/// [`REFUSAL_FRAME`] is the same sentence either way, and only the consequence
/// after it belongs to the entry point. `remedy` is the caller's because only the
/// shell's serve decision knows which causes no respelling can avoid.
pub(super) fn unserved_failure(reason: &str, remedy: Option<&str>) -> String {
    let message = format!(
        "{REFUSAL_FRAME}{reason}. The search did NOT run — this is not an empty \
         match set."
    );
    match remedy {
        Some(remedy) => format!("{message} {remedy}"),
        None => message,
    }
}

/// Aggregate telemetry fields derived purely from the per-member outcomes (shape +
/// counts). The row-level `served`/`reason` depend on runtime facts and are the caller's.
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
    /// A serve decision that runs the handed command unrewritten: `refusal` carries
    /// the Windows failure when the platform refuses it instead. The text handed to the
    /// engine is the shell's reading of the agent's line ([`crate::tools::shell::windows_line`]).
    fn not_rewritten(outcomes: Vec<GrepOutcome>, refusal: Option<String>) -> Self {
        Self {
            rewritten: None,
            outcomes,
            spec_files: Vec::new(),
            plan: None,
            refusal,
        }
    }

    /// Fold the per-member outcomes into the telemetry shape/count fields. `applied`
    /// gates served_count: when the rewrite wasn't applied (engine unavailable, spec
    /// too large, ReadOnly rejection, sentinel re-run) the engine served nothing.
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

/// Try to rewrite `command` into an engine-served equivalent, using the running
/// process's shell platform and engine probe. The engine preserves non-grep segments
/// verbatim, so the caller runs it in both ReadOnly and Full modes.
pub(super) fn try_serve_command(command: &str, workspace_root: &Path) -> GrepServe {
    serve_command(
        command,
        workspace_root,
        pinned_home().as_deref(),
        SHELL_PLATFORM,
        // Probe deferred to the point a member is servable, so a searchless call never spawns it.
        engine_available,
    )
}

/// The Windows refusal for a serve decision that runs the command it was handed: the
/// cause of the search it is known to carry, always `None` on unix.
fn refusal(platform: ShellPlatform, cause: Option<String>) -> Option<String> {
    if platform == ShellPlatform::Windows {
        cause
    } else {
        None
    }
}

/// A serve decision demoted to the command it was handed: no rewrite, every member
/// marked not-served with `reason` (the row's served field must match the ACTUAL
/// execution), and the platform's refusal — the command is known to carry a search.
fn demoted(platform: ShellPlatform, mut outcomes: Vec<GrepOutcome>, reason: &str) -> GrepServe {
    all_not_served(&mut outcomes, reason);
    let refusal = refusal(platform, Some(reason.to_string()));
    GrepServe::not_rewritten(outcomes, refusal)
}

/// The single routing decision for one command: a served rewrite, or the command
/// it was handed, unchanged. The platform and the engine probe are explicit
/// parameters so both platforms' routing — and every demotion path — is drivable
/// from any host's unit-test lane.
///
/// The probe is a closure, not a result: answering it spawns the full binary, so it
/// is asked only once a complete analysis has left something to serve. On unix an
/// unserved member keeps falling back to the real `grep`, so only a produced rewrite
/// ever changes what runs; on Windows a command carrying a grep-family invocation is
/// either fully served or refused ([`GrepServe::refusal`]).
///
/// `home` is the shell's pinned `$HOME`; `None` means nothing that could resolve `~`
/// is served — except on Windows, where nothing expands `~` and no decision reads it.
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
        // `resolve_operand`/`resolve_cd` never read `home` on Windows, so the empty
        // placeholder is never consulted (pinned by `refusal_pins`'s windows verdict table).
        None if windows => Path::new(""),
        None => return GrepServe::not_rewritten(Vec::new(), None),
    };
    // Single-file members are served on Windows: there is no host grep to be faster
    // than, so the unix perf gate would only lose the serve.
    let allow_single = windows;
    let Analyzed {
        specs,
        shapes,
        segments,
        outcomes,
        root_cwd,
        unserved,
    } = match analyze_command(command, workspace_root, home, platform, allow_single) {
        Ok(analyzed) => analyzed,
        Err(fail) => {
            let AnalyzeFailure {
                reason,
                mut outcomes,
                unserved,
            } = fail;
            // The refusal cause: the member cause when a member exists, else — Windows
            // only — the tolerant command-position scan, which decides whether a
            // structural abort (`;`, heredoc, an unreadable line, a trailing connector)
            // even carried a search.
            let cause = unserved.or_else(|| {
                (windows
                    && !matches!(reason, Fallback::NoGrep)
                    && unreadable_line_search(command, platform))
                .then(|| reason.to_string())
            });
            // Pre-analysis structural failures carry no grep member — no telemetry row.
            // Otherwise the per-member outcomes are preserved so an all-skipped command
            // still records its shape, a would-be serve discarded by a structural abort
            // being demoted so the row's served field matches the ACTUAL execution.
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
    // A member the engine cannot serve is kept verbatim in the rewrite: on unix that
    // member is a real grep and the sibling serves still win; on Windows it would run
    // a program the platform does not have, so the whole command is refused.
    if let Some(cause) = unserved {
        return demoted(platform, outcomes, &cause);
    }
    if !engine_ready() {
        return demoted(platform, outcomes, "engine unavailable");
    }
    // Serialized once per member: the same JSON is measured against the payload bound
    // and rendered into the fragment below.
    let jsons: Vec<String> = specs.iter().map(spec_json).collect();
    let limit = match platform {
        ShellPlatform::Unix => MAX_SPEC_JSON,
        ShellPlatform::Windows => MAX_SPEC_FILE,
    };
    if jsons.iter().any(|json| json.len() > limit) {
        return demoted(platform, outcomes, "spec exceeds payload limit");
    }
    // Only now — every analysis decision made, the whole command servable — is the
    // rewrite rendered, so the Windows scratch files cannot be created and abandoned
    // by a later demotion. Without this process's image there is no rewrite, so a
    // runtime doubt demotes the command (unix runs the original, Windows refuses).
    let rendered = std::env::current_exe()
        .map(|path| path.to_string_lossy().into_owned())
        .map_err(|e| Fallback::Handoff(format!("no own executable path: {e}")))
        .and_then(|exe| join_rewritten(&segments, &specs, &jsons, platform, &exe, &root_cwd));
    let Joined {
        text: rewritten,
        spec_files,
        plan,
    } = match rendered {
        Ok(rendered) => rendered,
        Err(reason) => {
            tracing::debug!(command = command, %reason, "grep engine: fallback");
            return demoted(platform, outcomes, &reason.to_string());
        }
    };
    // Both served forms log at DEBUG: gate-relaxation volume telemetry is too noisy
    // for INFO, and the dedicated table is the source of truth.
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
        // `analyze_command` hands back `Err` when it served no member.
        rewritten: Some(rewritten),
        outcomes,
        spec_files,
        plan,
        refusal: None,
    }
}

/// Best-effort `GrepOutcome` for a per-segment grep skip kept verbatim in the rewrite.
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

/// Canonical telemetry flag order, used by `lightweight_scan` to render a skipped
/// member's flag string. `flags_surface` follows the same relative order by
/// convention only, with one deliberate divergence: it folds before/after into `C`
/// when both are set, while `lightweight_scan` emits separate `A`/`B` letters.
const FLAG_ORDER: &[char] = &[
    'n', 'i', 'v', 'w', 'x', 'a', 'h', 'H', 's', 'r', 'o', 'z', 'c', 'l', 'm', 'C', 'A', 'B',
];

/// Telemetry scan of a command/segment for a skipped grep: `-r`/`-R`, a rough
/// operand count, and the short-flag letters in canonical order. Not a full parse.
/// Letters keep their case so `-h`/`-H`, `-c`/`-C`, `-A`/`-B`/`-a` stay
/// distinguishable, matching `flags_surface`; `-L` is dropped (absent from `FLAG_ORDER`).
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

/// Whether a served spec exercises the parallel recursive walk (so parity comparisons
/// relax to sorted line-sets: cross-file worker ordering is non-deterministic, in-file
/// ordering stays stable). Only the macOS-gated matrix uses this.
#[cfg_attr(
    not(all(test, target_os = "macos")),
    expect(dead_code, reason = "used only by the macOS-gated parity matrix")
)]
fn spec_uses_parallel_walk(spec: &EngineSpec) -> bool {
    spec.flags.r
        && spec
            .operands
            .iter()
            .any(|op| std::fs::metadata(&op.resolved).is_ok_and(|m| m.is_dir()))
}

/// The shell tool's pinned `$HOME` — the child shell's synthetic baseline
/// (`UserDirs`), never the ambient daemon env.
fn pinned_home() -> Option<PathBuf> {
    directories::UserDirs::new().map(|d| d.home_dir().to_path_buf())
}

/// Serialize one spec for the hand-off, so the payload bound and the rendered fragment
/// are measured on the same string.
fn spec_json(spec: &EngineSpec) -> String {
    serde_json::to_string(spec).expect("spec serializes")
}

/// Remove engine stream-size marker line(s) from stderr; returns the last reported
/// byte count. Any line containing the token goes (a non-marker line that literally
/// carries it is lost too — accepted), and `;`-joined commands emit several markers of
/// which only the last is logged. The marker must never reach the agent's stderr.
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
        // Reuses locate_line (grep_searcher-mirroring): the strip must apply the same
        // line-boundary semantics.
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

/// Process-lifetime mark that the engine answered a probe: only THAT is cached, since
/// a probe spawns the full binary. A failure must heal — on Windows the result is
/// load-bearing, and a one-shot failure (a spawn refused under resource pressure, the
/// binary caught mid-swap during a self-update) would otherwise turn every later
/// search into a hard refusal. A persistently stale binary then pays one probe per call
/// with something to serve, with the engine's version check and the parent's
/// lock-message handling as that case's other safety nets.
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
    // `creation_flags` is a Windows-only API, so this has to be a `cfg`; the flag it
    // carries is the window guarantee documented in `tools::shell::tree`.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        probe.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    }
    probe.status().is_ok_and(|s| s.success())
}

/// One served member rendered for both of its consumers: the shell fragment the
/// rewrite carries, and the argv that spawns the same program when the runner runs it
/// itself ([`super::plan`]'s `Run::Own`).
struct ServedMember {
    /// The fragment as the shell must run it — cmd.exe's quoting, or `sh`'s on unix.
    text: String,
    /// The argv after the image path, on the platform whose runner spawns the member
    /// itself. `None` on unix, where the shell runs the fragment and no plan exists.
    argv: Option<Vec<String>>,
}

/// Render one served grep member into the shell fragment that runs the engine: the
/// current executable, the engine verb and the spec, plus the member's own redirect
/// tokens kept verbatim.
///
/// The spec reaches the engine by argv on unix and by scratch file on Windows
/// ([`windows::write_spec_file`]): cmd.exe caps its command line at 8191 UTF-16 code
/// units and re-parses it, so the JSON — the agent's pattern and operands — cannot
/// ride on it. That scratch file is pushed onto `files` (the caller's removal list)
/// and is the same file the returned argv names.
///
/// `exe` and `json` are arguments so the one machine-dependent part of a rewrite is
/// drivable from a test, and so the caller's measured string is not re-serialized.
fn render_served(
    json: &str,
    redirects: &[String],
    platform: ShellPlatform,
    exe: &str,
    files: &mut Vec<PathBuf>,
) -> Result<ServedMember, Fallback> {
    let mut fragment = String::new();
    let mut argv = None;
    match platform {
        ShellPlatform::Unix => {
            fragment.push_str(&shell_quote(exe));
            fragment.push(' ');
            fragment.push_str(ENGINE_VERB);
            fragment.push(' ');
            fragment.push_str(&shell_quote(json));
        }
        ShellPlatform::Windows => {
            // Quote and allocate before writing: a refused argument must not leave a
            // scratch file behind. The allocated name goes onto `files` before the write,
            // or a failed/partial write (ENOSPC/EIO) leaks the file it created.
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
            // The runner's own spelling of the same call: argv, so cmd.exe's reading is not re-run.
            argv = Some(vec![
                ENGINE_VERB.to_string(),
                windows::SPEC_FILE_FLAG.to_string(),
                path.to_string_lossy().into_owned(),
            ]);
        }
    }
    if !redirects.is_empty() {
        fragment.push(' ');
        fragment.push_str(&redirects.join(" "));
    }
    Ok(ServedMember {
        text: fragment,
        argv,
    })
}

/// Why a command was not served (telemetry + fail-closed decisions).
///
/// One enum for both platforms: the variants a platform cannot reach are never
/// constructed there (SingleFile from the unix-only perf gate, NotEstablished from
/// the runtime parity gate). [`resolve_cd`] returns it and [`plan`]'s `tracked_cd`
/// reads that function, which maps any failure to the plan's own refusal.
#[derive(Debug)]
pub(super) enum Fallback {
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
    /// An unquoted `>(…)`/`<(…)`: a word the shell builds, never a redirection, and
    /// nothing about it resolves statically — the member stays on the real `grep`.
    ProcessSubstitution,
    SingleFile,
    /// The locale in effect could not be proven to agree with the real search for this
    /// member's shape, so the member is kept verbatim and the real `grep` runs it. The
    /// word is the unmet capability's ([`parity::Need`]), for the telemetry reason.
    NotEstablished(&'static str),
    CdUntrackable,
    StdinOperands,
    StdinRecursive,
    SegmentEmpty,
    /// A member after a segment that may have exported a new environment: the parity
    /// verdict was measured for the environment the product hands commands, not for the
    /// one the line built for itself.
    EnvironmentChanged,
    /// A spelling the cmd.exe reading cannot follow: a `^`, a command group, an
    /// unbalanced quote, an empty member — all spellings the interpreter itself reads —
    /// or a word whose glued operator the token classifier keeps inside the word. The
    /// rendered cause names the engine as the reader, never the interpreter's syntax.
    CmdSyntax(String),
    /// A word of the member carries two or more `%`, so cmd.exe may expand a `%…%` pair
    /// before any program sees its argv, and quoting cannot prevent that
    /// ([`windows::has_percent_expansion`]) — hence the rendered cause states a count.
    Expansion(String),
    /// The engine could not be handed what it needs: this binary's path, or the scratch
    /// file the Windows transport carries the spec in.
    Handoff(String),
    /// A member the runner must spawn itself is in a shape the plan cannot run: an
    /// image spelling it cannot run as one plain call, a redirect it cannot apply, or a
    /// line the decomposition cannot represent ([`plan::build`]). The reason is
    /// [`plan::Refusal`]'s own, so a second rendering cannot drift from it.
    PlanRefusal(String),
}

impl From<plan::Refusal> for Fallback {
    fn from(refusal: plan::Refusal) -> Self {
        Fallback::PlanRefusal(refusal.to_string())
    }
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
            Fallback::NotEstablished(cap) => {
                write!(f, "locale parity not established ({cap})")
            }
            Fallback::CdUntrackable => write!(f, "cd untrackable"),
            Fallback::EnvironmentChanged => write!(f, "the line changes the environment first"),
            Fallback::StdinOperands => write!(f, "stdin with operands"),
            Fallback::StdinRecursive => write!(f, "stdin with -r"),
            Fallback::SegmentEmpty => write!(f, "empty command or pipeline member"),
            Fallback::CmdSyntax(s) => write!(f, "cmd.exe spelling the engine cannot read: {s}"),
            Fallback::Expansion(s) => write!(f, "two `%` in {s}"),
            Fallback::Handoff(s) => write!(f, "hand-off refused: {s}"),
            Fallback::PlanRefusal(s) => write!(f, "{s}"),
        }
    }
}

/// A wholesale `analyze_command` failure: the original runs (unix) or the call is
/// refused (Windows). `reason` names the cause, `outcomes` preserves the decisions
/// collected before the abort, and `unserved` is the first unserved search member.
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

/// Demote every analyzed grep member to not-served with `reason` (the telemetry fields
/// must reflect the ACTUAL execution); per-member shape fields are preserved.
fn all_not_served(outcomes: &mut [GrepOutcome], reason: &str) {
    for o in outcomes {
        o.served = false;
        o.reason = reason.to_string();
    }
}

/// One member of the rewritten command: the original text verbatim, or a served grep
/// member (index into the analyzed specs) with its redirect tokens preserved.
enum OutSegment {
    Verbatim(String),
    Served { spec: usize, redirects: Vec<String> },
}

/// Analyze output: one spec per served grep, the shape of every served multi-member
/// pipeline (volume telemetry), the rewritten members with their original connectors
/// (rendered by [`join_rewritten`], so no scratch file precedes servability), and the
/// per-grep serve decisions.
struct Analyzed {
    specs: Vec<EngineSpec>,
    shapes: Vec<(usize, String, bool)>,
    segments: Vec<(OutSegment, String)>,
    outcomes: Vec<GrepOutcome>,
    /// The cwd the line starts in (the canonicalized workspace root), carried rather
    /// than recomputed so the plan's first step and the analyzer's cwd cannot disagree.
    root_cwd: PathBuf,
    /// Windows only (`None` on unix, where an unserved member just falls back): the
    /// cause of the FIRST member the engine cannot serve — see [`SEARCH_OWNING_VERBS`]
    /// for the one shape left unserved-but-allowed. A non-`None` value refuses the whole
    /// command: a verbatim `grep` would run a program this platform does not have.
    unserved: Option<String>,
}

/// Record the first member this platform cannot serve. Windows only, where the FIRST
/// cause wins and names the member the rewrite should start from; a no-op on unix.
fn note_unserved(platform: ShellPlatform, unserved: &mut Option<String>, reason: &str) {
    if platform == ShellPlatform::Windows && unserved.is_none() {
        *unserved = Some(reason.to_string());
    }
}

/// Analyze a full shell command: segment it, track cds, serve every grep member, keep
/// everything else verbatim. Returns the members or the first fallback reason.
#[expect(clippy::too_many_lines)] // per-segment serve/skip decision loop
fn analyze_command(
    command: &str,
    workspace_root: &Path,
    home: &Path,
    platform: ShellPlatform,
    allow_single: bool,
) -> Result<Analyzed, AnalyzeFailure> {
    // A `;` outside quotes is no separator to cmd.exe (several of its own commands read
    // it as an argument delimiter), so fail closed rather than serve a disputed member.
    if platform == ShellPlatform::Windows && windows::unquoted_semicolon(command) {
        return Err(Fallback::CmdSyntax("unquoted `;`".into()).into());
    }
    let stripped = strip_heredoc_bodies(command);
    if stripped != command {
        // strip_heredoc_bodies drops the body and terminator; rewriting around them would
        // leave a bare non-grep member reading inherited stdin — fail-closed.
        return Err(Fallback::Heredoc.into());
    }
    let segments = split_segments(&stripped, platform)?;
    if segments.is_empty() {
        return Err(Fallback::SegmentEmpty.into());
    }
    // No grep member anywhere: not a candidate. Before the shape checks so non-grep
    // pipelines don't pollute the pipeline-shape telemetry class.
    if !segments.iter().any(|(seg, _)| {
        let verb = first_word(seg);
        is_grep_verb(verb, platform)
            || segment_contains_grep(seg, &list_key(verb, platform), platform)
    }) {
        return Err(Fallback::NoGrep.into());
    }

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

    let root_cwd = canonical_or_lexical(workspace_root, platform);
    let mut cwd = root_cwd.clone();
    let mut rewritten: Vec<(OutSegment, String)> = Vec::new();
    let mut specs: Vec<EngineSpec> = Vec::new();
    let mut shapes: Vec<(usize, String, bool)> = Vec::new();
    // A shell-level `exec` stderr redirect seen so far (greps after it lose the marker).
    let mut exec_redirects_stderr = false;
    // Members after a served grep are preserved verbatim — real tools on a byte-identical
    // uncapped stream; only the first grep in a pipeline is served.
    let mut tail_preserved = vec![false; n];
    // Per-grep serve decisions (telemetry) for every analyzed grep member; the first
    // per-segment skip reason is kept so an all-skipped command falls back with a cause.
    let mut outcomes: Vec<GrepOutcome> = Vec::new();
    let mut first_skip_reason: Option<Fallback> = None;
    // Windows only: the cause of the first member the engine cannot serve.
    let mut unserved: Option<String> = None;
    // Group depth across segments (unquoted `(`/`{` openers minus closers): greps inside
    // an open compound group are never served, even when the splitter separated them from
    // the opener (`( cd d && grep … )`).
    let mut group_depth = 0isize;
    // Set by a segment that may have exported a new environment for the rest of the line
    // (see [`changes_the_environment`]); a grep after it runs under an environment the
    // parity gate does not cover.
    let mut env_may_have_changed = false;

    for (idx, (seg, conn)) in segments.iter().enumerate() {
        let in_group = group_depth > 0;
        let delta = group_delta(seg);
        // Preserved tails bypass all analysis: second greps and grep introducers in tail
        // positions are real tools on that stream.
        if tail_preserved[idx] {
            // A second grep is such a preserved member, consuming the engine's stream.
            // Windows has no program to run there, so the command is refused.
            if grep_family(first_word(seg), platform).is_some() {
                note_unserved(platform, &mut unserved, &Fallback::NestedGrep.to_string());
            }
            rewritten.push((OutSegment::Verbatim(seg.clone()), conn.clone()));
            group_depth = (group_depth + delta).max(0);
            continue;
        }
        let verb = first_word(seg);
        // Every verb list is read through the platform's own key, so a spelling classifies alike.
        let key = list_key(verb, platform);
        // The environment the earlier segments left behind is what this member runs
        // under; this segment's own change applies to what follows it — but only when the
        // segment is not handed to a pipe, since a pipeline member runs in a subshell of
        // its own (`export LC_ALL=C | grep …` leaves the sibling and what follows intact).
        let env_changed = env_may_have_changed;
        if !matches!(conn.as_str(), "|" | "|&") && changes_the_environment(seg, platform) {
            env_may_have_changed = true;
        }
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
            // The first grep in a pipeline is served wherever it sits; a non-first member
            // is fed by the producer's stdout. Later greps are preserved verbatim.
            let ctx = PipelineCtx {
                piped: pstart[idx] != pend[idx],
                stdin_fed: pstart[idx] != idx,
                // A shell-level `exec 2>&1` before this merges the marker into the captured
                // stdout, where the parent's strip cannot reach it.
                marker_ok: !exec_redirects_stderr,
                env_changed,
            };
            // Grep inside an open compound group: never served.
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
                        // Served-pipeline shape for volume telemetry, spanning the whole pipeline.
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
                    // Per-grep fallback: skip this segment, record the skip and continue — a
                    // servable sibling is still served (unix; on Windows the skipped member
                    // refuses the whole command — see `Analyzed::unserved`).
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
        // Non-grep member: verbatim, and a producer in a pipeline. Indirect/compound
        // invocations (xargs grep, sh -c, sudo, for bodies) are skipped per-segment rather
        // than falling the whole command back; `git grep`/`docker run … grep` are the
        // introducer's own search, exempt from the Windows refusal ([`SEARCH_OWNING_VERBS`]).
        let compound = is_compound_segment(seg, platform);
        let carries_grep = segment_contains_grep(seg, &key, platform);
        if compound || carries_grep {
            // A telemetry skip is recorded only when the segment carries a grep member: bare
            // construct keywords must not inflate the grep/skipped counts.
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
            // `exec 2>&1`/`exec 2>…`/`exec &>…` moves the shell's stderr for the rest of the
            // command, so the stream-size marker is suppressed for following greps (it would
            // leak into stdout or a file). `exec 1>&2`/`exec 3>&2` are intentionally not
            // matched, digit-prefixed dups (`exec 12>&1`) match `2>` and suppress unnecessarily
            // (telemetry loss only), and `FOO=1 exec 2>&1` / `\exec 2>&1` escape the check.
            exec_redirects_stderr = true;
        }
        rewritten.push((OutSegment::Verbatim(seg.clone()), conn.clone()));
        group_depth = (group_depth + delta).max(0);
    }

    // Served iff at least one grep was served; otherwise the command falls back wholesale
    // with the first skip reason naming the cause, and whether it then runs at all is
    // [`GrepServe::refusal`]'s. The per-member outcomes are preserved for the caller.
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
        root_cwd,
        unserved,
    })
}

/// One analysis rendered for both of its consumers: the command line the shell was
/// going to run, and — where a call of this service's own image is the runner's rather
/// than the shell's — the plan it executes instead ([`super::plan`]).
struct Joined {
    /// The rewrite as one command; the joining is [`join_rewritten`]'s.
    text: String,
    /// Scratch spec files the rewrite names, for the caller to remove.
    spec_files: Vec<PathBuf>,
    /// The runner's own decomposition of the same members: `Some` only when a rewrite was
    /// produced on the platform that needs one. A demotion produces no plan, and on
    /// Windows one carrying a cause is refused rather than run ([`GrepServe::refusal`]).
    plan: Option<Plan>,
}

/// Join the rewritten members with their own connectors into one command, verbatim on
/// both platforms: a newline connector can only come from the unix core and is emitted
/// verbatim because `sh -c` runs that rewrite. The rendered spec files are returned to
/// the caller — on Windows they are created here, the first point at which the whole
/// command is known to be servable — and a member whose hand-off fails takes the files
/// already written by its predecessors with it, since the command is demoted.
///
/// The same traversal collects the [`Plan`]'s members, so the text and the plan are two
/// renderings of ONE per-member decision and cannot drift apart; a plan that failed to
/// build demotes the command through the same `Fallback` arm a rendering failure does.
///
/// `exe` is this binary's path (a parameter for [`render_served`]'s reason) and
/// `root_cwd` is the analyzer's initial cwd — the cwd of every member up to the first
/// served one.
fn join_rewritten(
    segments: &[(OutSegment, String)],
    specs: &[EngineSpec],
    jsons: &[String],
    platform: ShellPlatform,
    exe: &str,
    root_cwd: &Path,
) -> Result<Joined, Fallback> {
    let mut out = String::new();
    let mut files = Vec::new();
    let mut members: Vec<(plan::Member, String)> = Vec::new();
    for (i, (seg, conn)) in segments.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        match seg {
            OutSegment::Verbatim(text) => {
                out.push_str(text);
                members.push((plan::Member::Shell { text: text.clone() }, conn.clone()));
            }
            OutSegment::Served { spec, redirects } => {
                match render_served(&jsons[*spec], redirects, platform, exe, &mut files) {
                    Ok(rendered) => {
                        out.push_str(&rendered.text);
                        if let Some(argv) = rendered.argv {
                            members.push((
                                plan::Member::Own {
                                    argv,
                                    redirects: redirects.clone(),
                                    cwd: PathBuf::from(&specs[*spec].cwd),
                                },
                                conn.clone(),
                            ));
                        }
                    }
                    Err(reason) => {
                        discard_spec_files(&files);
                        return Err(reason);
                    }
                }
            }
        }
        if i + 1 < segments.len() {
            out.push(' ');
            out.push_str(conn);
        } else if conn == "&" {
            // The last member's connector is normally a no-op and is dropped with its
            // trailing position. A trailing `&` backgrounds the member instead: dropping it
            // would run the search in the foreground, under its own exit status. Windows
            // never produces one — its splitter fails closed on a trailing connector.
            out.push(' ');
            out.push_str(conn);
        }
    }
    // The plan is the Windows decomposition: taking a call of this service's own image
    // over is Windows-only, and it tracks each member's `cd` the way the analyzer did, so
    // text and plan cannot disagree about a directory either.
    let plan = match platform {
        ShellPlatform::Windows => {
            match plan::build(&members, Path::new(exe), root_cwd) {
                Ok(plan) => Some(plan),
                // A refused line runs nothing, so its already-written scratch files must go too.
                Err(refusal) => {
                    discard_spec_files(&files);
                    return Err(Fallback::from(refusal));
                }
            }
        }
        ShellPlatform::Unix => None,
    };
    Ok(Joined {
        text: out,
        spec_files: files,
        plan,
    })
}

/// Remove scratch spec files, best-effort — the single home of that policy: the engine
/// calls it when a hand-off abandons a rewrite, so a refused Windows search leaks none,
/// and the shell's `SpecFiles` guard calls it on drop. A failing removal is the temp cleaner's.
pub(super) fn discard_spec_files(files: &[PathBuf]) {
    for path in files {
        let _ = fs::remove_file(path);
    }
}

/// Split a command into (segment, following-connector) pairs in `platform`'s own
/// spelling: `sh`'s `&&`, `||`, `;`, `|`, `|&`, `&`, `\n` or last; cmd.exe's `&&`,
/// `||`, `&`, `|` or last (a `\n` never arrives there — [`crate::tools::shell::windows_line`]
/// has respelled breaks). Quote- and substitution-aware per platform, heredoc bodies
/// already stripped. Empty segments before a connector (and a trailing `|`/`|&`/`||`/`&&`,
/// and a bare `&` on Windows) are shell syntax errors the rewrite would silently drop
/// into a VALID pipeline — fail-closed on the whole class. Blank lines and a trailing `&`
/// stay legal on unix.
fn split_segments(
    command: &str,
    platform: ShellPlatform,
) -> Result<Vec<(String, String)>, Fallback> {
    match platform {
        ShellPlatform::Unix => {
            super::segment_command(command, super::SegmentMode::Grep).ok_or(Fallback::SegmentEmpty)
        }
        // A line the cmd.exe model does not cover — a caret, a command group, an
        // unbalanced quote, a line break, an empty member before a connector — leaves the
        // whole line unread; the empty member mirrors the unix segmenter's policy.
        ShellPlatform::Windows => windows::segment_command(command)
            .ok_or_else(|| Fallback::CmdSyntax("the whole line".into())),
    }
}

/// Net group-open delta of one segment's unquoted `(`/`{`/`)`/`}`, quote- and
/// escape-aware so pattern parens (`'a(b'`, `\(`) don't count; `${…}`/`$(…)`/`$((…))`
/// cancel, so a group spanning segments (`( cd d && grep … )`) counts correctly. An
/// unbalanced unquoted `(`/`{` in a pattern or operand word leaves the depth positive
/// and wrongly skips a later top-level sibling — fail-closed, so the output stays right.
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

/// The grep family `verb` names on `platform`: the literal `grep`/`egrep`/`fgrep` on
/// unix, and on Windows whatever cmd.exe would dispatch to ([`windows::grep_verb`]).
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
/// case-insensitively and through an `.exe` suffix, so `CMD /C`, `cmd /c` and `cmd.exe /c`
/// must classify alike — a raw first word would let another spelling slip past every
/// list (neither served nor refused, reaching the agent as "not recognized"). unix
/// verbs are the words themselves; only the grep family is folded ([`grep_family`]).
/// A word the platform would re-read folds to the empty key, which matches no entry.
fn list_key(verb: &str, platform: ShellPlatform) -> String {
    match platform {
        ShellPlatform::Unix => verb.to_string(),
        ShellPlatform::Windows => windows::verb_key(verb).unwrap_or_default(),
    }
}

/// True when the segment's first word names a cwd-changing builtin: `cd` alone on unix,
/// cmd.exe's whole family on Windows (`cd`/`chdir`, `pushd`, `popd`), read through
/// [`list_key`] like every other list — the grouping the read-only guard's
/// `INTERNAL_VERBS` uses. Tracking only `cd` would leave the cwd stale, which is a wrong
/// directory rather than a missing one: unix refuses at runtime through the engine's cwd
/// gate, Windows refuses the line in [`plan`]. [`plan`]'s `tracked_cd` asks this too, and
/// [`is_cd_spelling`] is the other half for the spellings this key does not know.
pub(super) fn is_cd_segment(verb: &str, platform: ShellPlatform) -> bool {
    match platform {
        ShellPlatform::Unix => verb == "cd",
        ShellPlatform::Windows => matches!(
            list_key(verb, platform).as_str(),
            "cd" | "chdir" | "popd" | "pushd"
        ),
    }
}

/// The other half of that reading: a spelling of cmd.exe's cwd family [`is_cd_segment`]
/// does not follow, for a caller that must refuse rather than track. cmd ends an
/// internal command's name at the first character that cannot be part of it, so `cd..`,
/// `cd\Users`, `cd/d`, `@cd` and `cd.txt` are the builtin with target or switch fused,
/// while `cdrecord` and `C:\ws\cd.exe` are not. The whole family, because the tracking
/// follows all of it. Windows-only, like every cmd.exe reading.
#[must_use]
pub(super) fn is_cd_spelling(word: &str, platform: ShellPlatform) -> bool {
    if platform != ShellPlatform::Windows {
        return false;
    }
    let word = word.strip_prefix('@').unwrap_or(word);
    let name = word.split(['\\', '/', '.']).next().unwrap_or("");
    matches!(
        name.to_ascii_lowercase().as_str(),
        "cd" | "chdir" | "popd" | "pushd"
    )
}

/// Command-introducer verbs whose argument list may contain a nested grep, matched
/// through [`list_key`] so `CMD /C` is the same introducer: a grep-family word there is
/// kept verbatim (never served, its semantics uncertain) and the segment recorded as a
/// telemetry skip — a servable sibling still serves on unix, while on Windows the word
/// refuses the whole command. `cmd`, `powershell` and `pwsh` are listed beside the unix
/// shells for the same reason; on unix they only add a `grep_telemetry` row.
///
/// The match is the word, not the meaning: a segment that merely *carries* a grep-family
/// word is refused on Windows even when it is no search (`cmd /c del grep`). That
/// fail-closed over-refusal is deliberate and the loud failure names it;
/// [`SEARCH_OWNING_VERBS`] is the one exemption and must be revisited with this list.
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

/// Compound-construct keywords: a segment starting with one (or with a `(`/`{` opener)
/// nests a construct, so no grep inside THAT segment is served; siblings still are.
const COMPOUND_KEYWORDS: &[&str] = &[
    "if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done", "case", "esac",
    "select",
];

/// The shell's variable builtins: a segment holding one in command position may have
/// exported a new value, so everything after it runs under an environment the parity
/// gate does not cover (see [`changes_the_environment`]).
const ENV_MUTATING_BUILTINS: &[&str] = &["export", "unset", "eval", "source", "."];

/// True when `seg` may have changed the environment the rest of the line runs under: an
/// assignment or one of [`ENV_MUTATING_BUILTINS`] in command position — or in the first
/// position of a compound body, which the splitter hands over with its keyword.
///
/// An assignment in command position counts whether or not it stands alone: alone
/// (`PATH=…; grep …`) it reaches the later member, because an exported name keeps its
/// export status when reassigned, and prefixed to a special builtin (`LC_ALL=C export
/// FOO=1; grep …`) that builtin leaves it in force. A plain prefixed form (`FOO=1 echo
/// hi; grep …`) scopes it to that command, so refusing it too is a deliberate
/// over-refusal (telling the two apart means tracking special builtins), as is a
/// subshell group whose change dies with the group.
///
/// Command position is what is read: a quoted spelling is read through
/// [`scan::strip_quoted_word`](super::super::scan), a leading run of backslashes is
/// dropped (`\export LC_ALL=C` runs the builtin), and an argument that merely looks
/// like an assignment (`grep -rn KEY=value .`) is not one. A member after such a
/// segment runs under the environment the line built for itself, where the real search
/// would have answered differently, so it is not served. Unix only: on Windows there is
/// no real search to diverge from.
///
/// Left as a recorded residual: a mutating word behind a wrapper keyword (`time export
/// …`, `! export …`), inside a construct body (`case …`), behind a wrapper word
/// (`command export …`) or a redirection (`>log export …`), with an escape inside it
/// (`ex\port`), and `set -a`/`set -o allexport` — each would mean reading redirects,
/// wrapper words, construct bodies and shell option state.
fn changes_the_environment(seg: &str, platform: ShellPlatform) -> bool {
    if platform == ShellPlatform::Windows {
        return false;
    }
    let mut words = seg
        .split_whitespace()
        .map(|word| crate::tools::shell::scan::strip_quoted_word(word).trim_start_matches('\\'));
    let first = words.next().unwrap_or_default();
    let second = words.next();
    let command_position =
        super::is_env_assignment(first) || ENV_MUTATING_BUILTINS.contains(&first);
    let body_position = is_compound_segment(seg, platform)
        && second.is_some_and(|word| {
            super::is_env_assignment(word) || ENV_MUTATING_BUILTINS.contains(&word)
        });
    command_position || body_position
}

/// True when a segment opens a compound construct (keyword prefix or a `(`/`{` opener):
/// such a command must not serve any grep. Keywords are read through [`list_key`].
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

/// True when a non-grep segment could contain a grep invocation we would not serve —
/// conservative: any grep-family word in an introducer segment, or an env-assignment
/// verb. `verb` is the segment's [`list_key`], so the lists are read as the platform dispatches.
///
/// The words are the platform's own: cmd.exe's tokenizer keeps a quoted argument as one
/// word, whose delivered text is then read as the command line a wrapped interpreter
/// receives — `cmd /c "grep -rn x ."` would otherwise be neither served nor refused.
/// unix keeps its own reading (a missed telemetry row, never a serve decision).
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

/// [`segment_contains_grep`]'s word test for cmd.exe's tokenizer output: any delivered
/// word — or the command line inside a quoted word — names the grep family. Recursion
/// strips at least one quote pair per round, so it terminates.
fn words_nest_grep(words: &[GrepWord]) -> bool {
    words.iter().any(|w| {
        is_grep_verb(&w.value, ShellPlatform::Windows)
            || (w.value != w.raw
                && windows::tokenize(&w.value).is_some_and(|inner| words_nest_grep(&inner)))
    })
}

/// Programs that own the search they run: a `grep` word in their argument list is data,
/// not a member. `ssh` is here from the other side — its argument list runs on another
/// machine. A command whose ONLY grep word sits in one is left unserved but allowed on
/// Windows, where the platform refusal applies to members the engine could serve.
/// Matched through [`list_key`]. `find` is deliberately NOT here: `find … -exec grep …`
/// would run a `grep` Windows lacks — the `\;` spelling reaches the platform only
/// because its unquoted `;` aborts the analysis, which is refused only when a
/// command-position search is visible (this spelling shows none). Accepted residual:
/// what those programs report is never this engine's empty match set.
const SEARCH_OWNING_VERBS: &[&str] = &["git", "docker", "kubectl", "podman", "ssh"];

/// True when `command` carries a grep-family invocation in command position, by a
/// deliberately tolerant scan: split on `&` and `|` plus, over-wide on purpose, a line
/// break and `;`, with no quote handling, then strip the leading whitespace and any
/// group opener (`(`/`{`) off each fragment and test its first word through the
/// analyzer's predicate. The opener is stripped rather than split on, so `(grep …)` is
/// still read as the grep it is while `echo (grep is a tool)` stays an echo.
///
/// It exists only to decide whether a line the cmd.exe model could not read AT ALL (a
/// caret, a group, an unbalanced quote, a `;`, a trailing connector) should be refused
/// on Windows: such a line aborts before any member exists. It errs toward refusing,
/// while leaving `git commit -m "grep fix"`, `echo grep` and `git grep x` alone.
///
/// Being quote-blind it can lift a grep word out of a quoted string and blame a line
/// that carries no search (`echo "a; grep x" && cd /x`). Fail closed, and accepted: a
/// quote-aware scan would have to trust the quotes of a line it could not read.
fn unreadable_line_search(command: &str, platform: ShellPlatform) -> bool {
    command.split(['&', '|', ';', '\n']).any(|fragment| {
        let fragment = fragment.trim_start_matches(['(', '{', ' ', '\t']);
        grep_family(first_word(fragment), platform).is_some()
    })
}

/// Resolve a literal cwd-family segment (see [`is_cd_segment`]) against the tracked cwd,
/// returning the new canonical cwd; only statically-resolvable targets track.
///
/// The grammar is the platform's: `sh`'s option scan and `$HOME` target on unix;
/// cmd.exe's `/d`-only, nothing-changes-on-bare-`cd` reading (`~` is an ordinary name)
/// on Windows, where `popd` (its target lives on the pushed stack) and a bare `pushd`
/// (pushes, then that drive's root) are fail-closed with [`Fallback::CdUntrackable`],
/// while `pushd <target>` tracks like `cd <target>`. Two cmd spellings stay outside: a
/// glued switch (`cd/d C:\ws`, refused earlier by [`is_cd_spelling`]) and a
/// drive-relative target (`cd C:foo`). A wrong tracked cwd is a wrong answer, not a
/// missing one, so the command is refused instead of served.
///
/// [`plan`]'s `tracked_cd` reads a `cd` through this too, so a member after a `cd` runs
/// in the directory this tracking named.
pub(super) fn resolve_cd(
    segment: &str,
    cwd: &Path,
    home: &Path,
    platform: ShellPlatform,
) -> Result<PathBuf, Fallback> {
    if platform == ShellPlatform::Windows {
        // cmd.exe's own word reading, not a whitespace split: a quoted target keeps its
        // inner spaces, and [`windows::cd_scan`] judges the spelling.
        let words = windows::tokenize(segment).ok_or(Fallback::CdUntrackable)?;
        let Some((verb, rest)) = words.split_first() else {
            return Err(Fallback::CdUntrackable);
        };
        let verb = list_key(&verb.raw, platform);
        let rest: Vec<&str> = rest.iter().map(|w| w.raw.as_str()).collect();
        // The stack is what `popd` reads and this model does not keep.
        if verb == "popd" {
            return Err(Fallback::CdUntrackable);
        }
        let (target, next) = match windows::cd_scan(&rest) {
            CdScan::Target(target, next) => (target, next),
            // A bare `cd`/`chdir` prints the cwd and changes nothing; a bare `pushd` is not
            // modelled — fail-closed.
            CdScan::Bare => {
                if verb == "pushd" {
                    return Err(Fallback::CdUntrackable);
                }
                return Ok(cwd.to_path_buf());
            }
            // A switch-shaped word (`/x`) or a spelling cmd reads differently — fail-closed.
            CdScan::BadOption => return Err(Fallback::CdUntrackable),
        };
        if rest.get(next).is_some() {
            // Extra operands are shell-dependent — fail-closed.
            return Err(Fallback::CdUntrackable);
        }
        if windows::has_percent_expansion(target) {
            return Err(Fallback::CdUntrackable);
        }
        // A drive-relative target (`cd C:foo`) is the cwd of ANOTHER drive, which this
        // model does not keep: joining it onto this cwd would track a wrong directory.
        if windows::is_drive_relative(target) {
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

/// Resolve a shell path word: absolute paths pass through verbatim, anything else joins
/// onto the tracked cwd. Each platform reads its own absolute spelling.
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
/// (`std::path::absolute`, exact because every input is already absolute) and then to
/// the path as given. On Windows the verbatim (`\\?\`) prefix `fs::canonicalize` adds is
/// stripped, so tracked, operand and displayed paths carry cmd.exe's own spelling —
/// which is why the cwd gate needs [`windows::same_spelling`] rather than byte equality.
/// [`plan::build`]'s `root_cwd` starts here too, so both readings name one directory.
pub(super) fn canonical_or_lexical(p: &Path, platform: ShellPlatform) -> PathBuf {
    let canonical = fs::canonicalize(p)
        .or_else(|_| std::path::absolute(p))
        .unwrap_or_else(|_| p.to_path_buf());
    match platform {
        ShellPlatform::Unix => canonical,
        ShellPlatform::Windows => crate::util::strip_verbatim_prefix(&canonical),
    }
}

/// True when a raw shell word contains an expansion (`$`, backtick, `$(`) that cannot
/// be resolved statically (unquoted or double-quoted; single quotes suppress).
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

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
enum MatchMode {
    Basic,
    Extended,
    Fixed,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
// Lenient parse so a swapped binary's spec reaches the version check and execs the
// real grep in place (the sentinel is for a run that already wrote output).
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
    /// -c/-l select lines by count/name instead of printing them; -o becomes inert and
    /// context flags are accepted but ignored, so the flag-interaction gates don't apply.
    fn count_mode(&self) -> bool {
        self.c || self.l
    }

    /// Whether the search prints context lines: the only mode where the host command
    /// separates the blocks of different files with a `--` line.
    fn context_active(&self) -> bool {
        !self.count_mode() && (self.before > 0 || self.after > 0)
    }
}

/// Normalized single-string flag surface for a served grep (telemetry), in `GrepFlags`
/// field order except that before/after set together fold into `C` (`-B2 -A3` → "C").
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

/// A word from the grep segment: its unquoted value plus redirect metadata. Read by the
/// runner's own decomposition too ([`super::plan`]), so the word and its fields are
/// visible to siblings — the plan's cwd/argv/redirect reading is the same one.
#[derive(Clone)]
pub(super) struct GrepWord {
    pub(super) value: String,
    pub(super) raw: String,
    pub(super) redirect: bool,
    pub(super) needs_target: bool,
}

/// True when `token` already opens a redirect `sh` reads as one operator plus target (an
/// optional fd digit prefix included). A later operator byte inside such a token stays in
/// it: the rewrite re-emits the raw spelling verbatim, so the shell re-reads what it
/// read. The canonical full-token reading is [`super::scan::classify_shell_token`]; this
/// is the partial-token test the mid-word arms need, and the tokenizer's redirect pins
/// catch the drift.
fn opens_redirect(token: &str) -> bool {
    let rest = token.trim_start_matches(|c: char| c.is_ascii_digit());
    if rest.len() == token.len() {
        // No fd prefix: the token opens with the operator as written (`&>` is one operator).
        token.starts_with(['<', '>']) || token.starts_with("&>")
    } else {
        // An fd prefix: only `<`/`>` may follow it (`2&>log` is the word `2` plus `&>log`).
        rest.starts_with(['<', '>'])
    }
}

/// Tokenize a grep segment (after the verb): quote-aware split, unquote, redirect
/// classification delegated to the read-only guard's classifier (single source of truth),
/// in the platform shell's own reading.
fn grep_tokenize(segment: &str, platform: ShellPlatform) -> Result<Vec<GrepWord>, Fallback> {
    if platform == ShellPlatform::Windows {
        let words = windows::tokenize(segment)
            .ok_or_else(|| Fallback::CmdSyntax("unbalanced quotes".into()))?;
        // cmd.exe splits a word at an unquoted redirect operator glued to it while the shared
        // classifier reads only an operator that OPENS the token, so `x>out.txt` would be
        // served as one pattern with the redirect dropped — refused instead. An operator
        // glued to the *verb* (`grep>x f.txt`) never reaches this test: `grep>x` is no verb,
        // so the line carries no member. Windows only: the unix lane splits the glued
        // spelling out of the word itself.
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
            // An unquoted, unescaped `<`/`>` ends the word before it and starts a redirection,
            // exactly as `sh` reads it; the raw spelling then rides the rewrite verbatim.
            if c == '<' || c == '>' {
                // `>(…)`/`<(…)` is process substitution: a word the shell builds, never a
                // redirect this engine could resolve — the member stays on the real `grep`.
                if chars.peek() == Some(&'(') {
                    return Err(Fallback::ProcessSubstitution);
                }
                // POSIX IO_NUMBER: the digits before the operator belong to it only when the whole
                // word so far is digits (`2>log` redirects fd 2; `x2>log` is `x2` plus a redirect).
                let all_digits = current.bytes().all(|b| b.is_ascii_digit());
                if !all_digits && !opens_redirect(&current) {
                    flush(&mut current, &mut out)?;
                }
                current.push(c);
                // A glued `&` (`>&`, `<&`) needs no arm: `&` is never a word boundary here, so
                // it rides the ordinary path into the same token.
                continue;
            }
            // `&>` (and `&>>`) is one bash operator; the word before it stays part of the search.
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

/// Unquote a raw shell word into its literal value; any expansion (unquoted or
/// double-quoted `$`/backtick, ANSI-C `$'…'`) is a fallback. On Windows this is cmd.exe's
/// unquoting, which expands nothing — the `%…%` reading is the word scan's.
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

/// Member-side redirects that move stderr off the captured fd 2 (into the pipeline or a
/// file): the stream-size marker would corrupt the output or pollute the file, so stdin
/// serves with them skip it. `1>&2`/`>&2` and scratch-fd dups (`3>&1`) keep stderr
/// captured and are not matched; digit-fd dups like `12>&1` contain `2>` and suppress too
/// — fail-closed, telemetry loss only.
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
    /// The raw patterns as typed (before [`translate_pattern`]): the parity gate reads
    /// these, not the engine dialect's metacharacters.
    patterns: Vec<String>,
    engine_patterns: Vec<String>,
    operand_tokens: Vec<String>,
    redirects: Vec<String>,
    fallback_prefix: Vec<String>,
}

#[expect(clippy::too_many_lines)] // BSD-getopt flag table is inherently long
/// Parse a grep segment's words with BSD-getopt semantics: GNU-style permutation,
/// value-taking options consuming the rest of their token, last-wins -G/-E/-F, the macOS
/// option cluster, and the verified flag surface only — anything else falls back.
fn parse_grep_words(
    words: &[GrepWord],
    verb: &str,
    platform: ShellPlatform,
) -> Result<ParsedGrep, Fallback> {
    // Redirects (and their targets) are removed before grep sees the argv; keep them verbatim.
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
    // cmd.exe expands `%…%` before the program sees argv (a pattern, an `-e` value and a
    // filter spelling alike), so a word carrying a pair cannot be served with its literal
    // text ([`windows::has_percent_expansion`]). A lone `%` and every `!` are ordinary
    // here: `cmd /C` runs under pinned settings, `/V:OFF` among them. Redirects are exempt:
    // on unix the rewrite re-emits their tokens for `sh`; on Windows the runner applies
    // them itself ([`super::plan::apply_redirects`]), which is why `parse_redirects`
    // refuses a target carrying a `%…%` pair rather than expanding it.
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

    // Flag-surface validation: under -c/-l, -o is inert and context flags are accepted but
    // ignored, so their interaction gates don't apply.
    if flags.v {
        flags.o = false; // -v -o behaves as plain -v
    }
    if flags.v && flags.context_active() {
        return Err(Fallback::UnsupportedFlag("-v+context".into()));
    }
    if flags.o && flags.context_active() {
        return Err(Fallback::UnsupportedFlag("-o+context".into()));
    }
    if flags.o && flags.m.is_some() && !flags.count_mode() {
        return Err(Fallback::UnsupportedFlag("-m+-o".into()));
    }

    // Patterns: BSD splits each on newlines (add_pattern); an empty piece among several is
    // an empty-alternation fallback, while a single empty pattern matches all and serves.
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

    // Translation + compile validation happen here so most runtime fallbacks never reach dispatch.
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
    // `-o` + alternation: the engine (leftmost-first) diverges from BSD (leftmost-longest)
    // on match length — fail-closed. Fixed-string mode has no alternation operator, so a
    // literal `|` stays servable, and -o is inert under -c/-l.
    if flags.o
        && !flags.count_mode()
        && mode != MatchMode::Fixed
        && engine_patterns.iter().any(|p| has_alternation(p))
    {
        return Err(Fallback::UnsupportedFlag("-o+alternation".into()));
    }
    let matcher = build_matcher(&engine_patterns, mode, &flags)
        .map_err(|e| Fallback::CompileFailure(e.to_string()))?;
    // Empty-matchable patterns diverge between BSD and the engine; the single literal-empty
    // pattern is the one approved exception (matches everything).
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
        patterns: raw_patterns,
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

/// Translate a BSD-dialect pattern to the engine dialect (Rust regex syntax); anything
/// not provably equivalent falls back.
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

/// Escapes whose BSD-grep semantics differ from the engine (word anchors, PCRE-style
/// classes), shared by the ERE pre-check and the BRE translator so the reject list cannot
/// drift between the two paths.
fn bsd_escaped_class_reject(c: char) -> Option<&'static str> {
    match c {
        '<' | '>' => Some(r"\< \>"),
        's' | 'S' | 'w' | 'W' | 'd' | 'D' => Some(r"\s\w\d"),
        _ => None,
    }
}

/// ERE passes through, but reject constructs whose engine semantics differ from BSD
/// (backrefs and unknown escapes are caught by compile validation).
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

/// BRE → engine dialect. `\| \( \) \+ \? \{m,n\}` become metacharacters; unescaped
/// `| ( ) + ? { }` are literals (escaped); `\b` passes through; backrefs, `\<`/`\>`,
/// `\s`/`\w`/`\d` and unknown escapes fall back.
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

/// True when the pattern has unescaped alternation outside a character class. BSD is
/// leftmost-longest and the engine leftmost-first, so under `-o` it can diverge on length.
fn has_alternation(s: &str) -> bool {
    alternation_at(s, false)
}

fn has_top_level_alternation(s: &str) -> bool {
    alternation_at(s, true)
}

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

/// `-w` equivalence gate: `\<pat\>` ≡ `\b(?:pat)\b` only when every match starts and
/// ends with a word char — conservative: the translated pattern must begin and end with
/// one and carry no top-level alternation.
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

/// Matcher wrapper: BSD grep under C.UTF-8 treats any line containing invalid UTF-8 as a
/// silent non-match in every mode (-F/-E/-i/-v alike) — except in binary files, where a
/// NUL in the first 32 KiB switches grep to byte-oriented matching. The wrapper rejects
/// invalid haystacks when `validate_utf8` is set (text files). With a configured line
/// terminator the inner matcher engages the whole-buffer SIMD prefilter (grep-regex's
/// `fast_line_regex`/`find_candidate_line`); the `find_candidate_line` override
/// re-validates `Confirmed` lines, which the searcher reports without calling `find_at`.
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
        // Binary files are byte-oriented: the inner candidate/confirmed result stands.
        if !self.validate_utf8 {
            return Ok(Some(kind));
        }
        match kind {
            // The searcher verifies Candidate lines via `is_match` → `find_at`,
            // which already rejects invalid-UTF-8 lines.
            grep_matcher::LineMatchKind::Candidate(i) => {
                Ok(Some(grep_matcher::LineMatchKind::Candidate(i)))
            }
            // The searcher reports Confirmed lines without `find_at`, so an invalid-UTF-8
            // line would be a false match: re-validate and skip past it. Iterative (not
            // recursive) so a large non-UTF-8 file cannot overflow the worker stack.
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

/// Locate the line (terminator included) containing byte offset `pos`, mirroring the
/// private `grep_searcher::lines::locate` for a single point.
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

#[expect(
    clippy::struct_excessive_bools,
    reason = "independent facts about one member, each read on its own"
)]
#[derive(Clone, Copy)]
struct PipelineCtx {
    /// The member feeds a pipeline tail: its stdout must not be capped.
    piped: bool,
    /// Not the pipeline's first member: fed by the producer's stdout via stdin.
    stdin_fed: bool,
    /// The member's stderr still reaches the capture (no shell-level `exec` stderr merge);
    /// when false the stream-size marker is suppressed.
    marker_ok: bool,
    /// An earlier command may have changed the environment the member runs under (see
    /// [`changes_the_environment`]): it is not the environment the verdict was measured for.
    env_changed: bool,
}

/// Serve one grep segment: parse, translate, resolve operands, apply the serve-worthiness
/// gate, and build the spec plus its member-side redirect tokens. A non-first pipeline
/// member (`ctx.stdin_fed`) is served from the producer's stdin when it has no operands and
/// no -r. `allow_single` bypasses the single-file perf gate (set on Windows).
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
    // The segment starts with the verb; the parser sees only its arguments. The comparison
    // goes through the family normalizer (on Windows `GREP.EXE` is served as `grep`).
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
                // `grep -r pat` walks the cwd and ignores stdin — never stdin-feed it.
                return Err(Fallback::StdinRecursive);
            }
            operands.push(Operand {
                display: ".".into(),
                resolved: cwd.to_string_lossy().into_owned(),
                trailing_slash: false,
            });
        } else if ctx.stdin_fed {
        } else {
            return Err(Fallback::StdinMode);
        }
    } else {
        if ctx.stdin_fed {
            // Non-first grep with file operands (incl. "-") is a deliberate scope cut.
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

    // Serve gate: recursive walks (the expensive case) or multi-file invocations. A
    // single-file lookup stays on the real grep on unix (the host binary is faster), and
    // `allow_single` serves it on Windows. stdin-fed serves bypass the gate.
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

    // The line may have put this member in an environment of its own: the parity verdict is
    // the process environment's, so it is not served. After the perf gate, so telemetry
    // names the more specific reason first.
    if ctx.env_changed {
        return Err(Fallback::EnvironmentChanged);
    }

    // Locale-parity gate: the fast path may stand only where the battery established, under
    // the locale in effect, that this shape agrees with the real search (see [`parity`]).
    // After the perf gate so telemetry keeps the more specific reason first.
    if let Err(cap) = parity::serve_allowed(&parity::needs(&parsed, platform), platform) {
        return Err(Fallback::NotEstablished(cap));
    }

    let mut fallback = parsed.fallback_prefix.clone();
    // A `-`-prefixed operand would be re-parsed as a flag by the exec'd grep; `--` restores
    // operand status.
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
        // The marker would leak into the captured stdout (a shell-level `exec 2>&1` before
        // the member) or pollute a member-side stderr redirect — skip it then.
        report_stream_bytes: ctx.stdin_fed
            && ctx.marker_ok
            && !parsed.redirects.iter().any(|r| redirect_merges_stderr(r)),
    };

    Ok((spec, parsed.redirects))
}

/// Resolve one raw operand token into concrete operands (tilde, glob expansion,
/// relative-to-cwd). Quote/glob checks run on the token as typed: quoted/escaped `~` and
/// glob metacharacters are literal filenames to BSD grep and must not expand.
///
/// The `~` arms are unix-only (cmd.exe expands no `~`); the `%`/`!` reading is checked
/// once, in [`parse_grep_words`].
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
        // `~/` expands to home only when the raw token opens with an unquoted `~/`; `"~"/*.txt`
        // is a literal cwd-relative path, and the unquoted value would wrongly home-strip.
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

/// True when the raw token has glob metacharacters outside quotes/escapes, via
/// [`super::track_char_context`] (bash: glob chars inside single OR double quotes are
/// literal). Under cmd.exe only `"` quotes and there is no escape char, so the Windows
/// reading is the `windows` layer's.
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

/// Shell-style glob expansion for operand tokens: dotfile exclusion, bracket expressions,
/// no-match → empty (the caller falls back). Display strings match what the shell would
/// pass to grep; `~/…` home expansion is the caller's (unquoted leading `~/` only).
/// The walk root is the platform's own: `/` (or the cwd) with `/` components on unix, a
/// drive/UNC share/rooted separator (or the cwd) with `\` components on Windows.
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
    // The shell sorts glob expansions by its own collation while grep emits operands in
    // command-line order, so the order must match. That agreement is what the `Glob`
    // capability measures: it holds in a `C` locale and not in a collating one, so a
    // collating locale sends glob operands to the real grep.
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
            // Shell glob semantics: fnmatch with FNM_PERIOD (leading dot must match literally).
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

/// Join one glob component onto the display prefix it was matched under: `/` on unix (a
/// lone `/` prefix is the root), `\` on Windows (the prefix already carries its separator).
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

/// Entry point for the hidden `__grep-engine` subcommand (dispatched from `main()` before
/// instance-lock acquisition); `--probe` answers whether the binary still carries it
/// (self-update skew). Any runtime doubt hands the member back to the real grep: `exec`
/// in place on unix, and on Windows the reason plus the sentinel code the parent refuses on.
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
        // Self-update swapped the binary mid-run: nothing is written yet, so hand back.
        return cannot_serve(&spec, "spec version mismatch");
    }
    // `serve` itself reports panics: nothing-written panics return Err, post-output panics
    // return the sentinel. This is the last-resort net for panics outside its scope.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| serve(&spec)));
    match result {
        Ok(Ok(code)) if code != ENGINE_FAILED_EXIT => code,
        // Err only from pre-output checks (cwd mismatch, matcher build, an early panic).
        Ok(Err(reason)) => cannot_serve(&spec, reason),
        // A panic escaped `serve` or followed output: partial output may exist, so refuse.
        Ok(Ok(_)) | Err(_) => engine_failed(None),
    }
}

/// Report an engine failure the parent must treat as an unserved search: the one-line
/// `detail` (when there is one) and, where the parent reads it, [`ENGINE_REFUSAL_MARKER`]
/// on a line of its own, so a refusal survives a pipeline tail masking the sentinel status.
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

/// Read the spec JSON: a bare JSON argument on unix, or the scratch file named by
/// [`windows::SPEC_FILE_FLAG`] on Windows, where the command line cannot carry it. The
/// `Err` is the one-line reason printed before the sentinel exit.
fn read_spec(args: &[String]) -> Result<String, String> {
    match args {
        [flag, path, ..] if flag == windows::SPEC_FILE_FLAG => {
            fs::read_to_string(path).map_err(|e| format!("grep: engine: spec file {path}: {e}"))
        }
        [json, ..] => Ok(json.clone()),
        [] => Err("grep: engine: missing spec".to_string()),
    }
}

/// The real grep is about to produce the answer instead of the engine, because the engine
/// cannot serve this spec (cwd divergence, matcher build failure, protocol mismatch, a
/// panic before any output).
///
/// Unix: replace the process with the real grep — sound only while stdin was not yet
/// consumed (the pre-checks run before any stdin read) — and the PID/PGID are preserved,
/// so the shell tool's process-group timeout kill still applies.
///
/// Windows: cmd.exe resolves `grep` through `PATHEXT` and its own cwd, and the engine
/// cannot re-exec that exact program, so the reason goes to stderr and the sentinel exit
/// refuses the call. The `cfg` guards `exec`, not platform policy ([`SHELL_PLATFORM`]).
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

/// Replace the engine process with the real grep: PID/PGID preserved, so the process-group
/// timeout kill still applies. Unix-only — the Windows hand-back is the sentinel exit.
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

/// Serve one spec: verify the cwd, then process each operand in order with per-file grep
/// semantics (or read stdin for stdin-fed members). Returns the aggregate exit code.
///
/// `Err(reason)` means nothing was written and stdin was not consumed, so the caller can
/// hand the member back to the real grep. A panic after output started or after the stdin
/// head read returns `Ok(ENGINE_FAILED_EXIT)`: running grep then would append a false
/// result or read the drained pipe remainder.
fn serve(spec: &EngineSpec) -> Result<i32, &'static str> {
    let actual_cwd = std::env::current_dir().map_err(|_| "working directory unavailable")?;
    // The parent's tracked cwd and this process's must name the same directory. Unix
    // compares canonical spellings byte-exactly; Windows resolves this process's cwd
    // through the parent's canonicalization — path identity (case, separators, verbatim
    // prefix) plus the aliases `fs::canonicalize` follows (a junction or `subst` drive
    // would otherwise fail the gate for every member).
    let same_cwd = match SHELL_PLATFORM {
        ShellPlatform::Unix => actual_cwd == Path::new(&spec.cwd),
        ShellPlatform::Windows => windows::same_spelling(
            &canonical_or_lexical(&actual_cwd, ShellPlatform::Windows),
            Path::new(&spec.cwd),
        ),
    };
    if !same_cwd {
        // The cd chain diverged at runtime (e.g. a failing `cd`): the real grep is authentic.
        return Err("working directory diverged from the analyzed command");
    }
    let matcher = build_matcher(&spec.patterns, spec.mode, &spec.flags)
        .map_err(|_| "pattern could not be compiled")?;
    let mut out = Output::new(
        OutputSink::Stdout(io::BufWriter::with_capacity(16 * 1024, io::stdout())),
        StdoutDest::of(spec, stdout_is_pipe()),
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
        // A panic before anything was written or read: exec'ing the real grep is clean.
        Err(_) if out.written == 0 && !stdin_consumed.get() => {
            Err("engine panicked before producing output")
        }
        // A panic mid-search may have streamed output or consumed the producer's pipe, so
        // exec'ing grep would append a false result. Flush and return the sentinel.
        Err(_) => {
            out.finish();
            Ok(ENGINE_FAILED_EXIT)
        }
    }
}

/// Serve a spec writing into caller-owned buffers (testable in-process), with the stdin
/// reader injectable so tests never touch the process's real stdin.
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
    // `max` keeps Error over Match/NoMatch — any error yields exit 2 even after matches.
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

/// Read the 32 KiB binary-window head, returned truncated, with the NUL verdict.
fn read_binary_head<R: io::Read>(src: &mut R) -> io::Result<(Vec<u8>, bool)> {
    let mut head = vec![0u8; BINARY_WINDOW];
    let n = src.read(&mut head)?;
    head.truncate(n);
    let binary = head.contains(&0);
    Ok((head, binary))
}

/// Searcher with the spec's matching surface: context flags suppressed in count modes
/// ([`GrepFlags::context_active`]), -m capping matches (stopping input consumption).
fn build_searcher(spec: &EngineSpec) -> (grep_searcher::Searcher, bool) {
    let mut sb = grep_searcher::SearcherBuilder::new();
    sb.line_number(spec.flags.n)
        .invert_match(spec.flags.v)
        .binary_detection(grep_searcher::BinaryDetection::none())
        .heap_limit(Some(HEAP_LIMIT));
    if spec.flags.context_active() {
        sb.before_context(spec.flags.before)
            .after_context(spec.flags.after);
    }
    if let Some(m) = spec.flags.m {
        sb.max_matches(Some(m));
    }
    (sb.build(), spec.flags.count_mode())
}

/// Search a (possibly non-seekable) stream with the spec's BSD sink; `binary` is the
/// NUL-window verdict. The sink borrows `out`; the caller writes the stream-size marker
/// after this returns (its position among grep errors is irrelevant — the strip scans all).
fn search_stream<R: io::Read>(
    spec: &EngineSpec,
    matcher: &grep_regex::RegexMatcher,
    display: &str,
    show_prefix: bool,
    binary: bool,
    input: &mut R,
    out: &mut dyn LineOut,
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
            block_pending: true,
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

/// Serve a stdin-fed grep member with BSD binary semantics: a 32 KiB NUL-sniff buffer
/// filled by one partial read (a pipe yields only its available bytes), the
/// "(standard input)" display name, filters ignored, per-line flush, and -m stopping the
/// read (grep-searcher's max_matches).
fn serve_stdin<R: io::Read>(
    spec: &EngineSpec,
    matcher: &grep_regex::RegexMatcher,
    out: &mut dyn LineOut,
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
    // Non-seekable stream: buffer the head, then the remainder (as for FIFOs and /dev/*).
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

/// Process one operand (explicit file/dir or walk root) with BSD symlink and error semantics.
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
    // Explicit file operands are always searched, subject to the BSD include/exclude filters.
    if !file_allowed_by_filters(display, spec, SHELL_PLATFORM) {
        return OperandResult::NoMatch;
    }
    search_file(path, display, spec, matcher, out, show_prefix)
}

/// BSD fnmatch semantics for --include/--exclude: a pattern matches the basename OR the
/// full traversal path, `*` crosses `/`, anchored full-string, and the LAST matching
/// pattern in command-line order decides.
fn file_allowed_by_filters(display: &str, spec: &EngineSpec, platform: ShellPlatform) -> bool {
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

/// The basename of a displayed path in the platform's separator reading (`\` separates only
/// on Windows).
fn display_basename(display: &str, platform: ShellPlatform) -> &str {
    match platform {
        ShellPlatform::Unix => display.rsplit('/').next().unwrap_or(display),
        ShellPlatform::Windows => display.rsplit(['/', '\\']).next().unwrap_or(display),
    }
}

/// Shell glob matching, `period` standing for `FNM_PERIOD` (a leading dot must be matched
/// by a literal dot): the host `fnmatch` on unix — bash globbing passes `FNM_PERIOD`, BSD
/// grep's filters pass no flags — and the pure-Rust [`windows::fnmatch`] on Windows. The
/// two are pinned differentially by `windows_fnmatch_parity`, which runs on a unix host.
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
        // No host matcher exists off unix; the unix platform is a test-lane value there.
        false
    }
}

/// fnmatch without FNM_PATHNAME — exactly what BSD grep calls: `*`/`?` cross `/`, `[...]`
/// classes (incl. POSIX names), anchored full-string.
fn fnmatch(pattern: &str, s: &str, platform: ShellPlatform) -> bool {
    fnmatch_flags(pattern, s, false, platform)
}

/// Recursive walk with rg-default exclusions (hidden, gitignore, .ignore, git exclude,
/// global gitignore) plus BSD include/exclude filters and symlink skipping; explicit
/// operands never pass through here.
///
/// The SEARCH runs on multiple worker threads: each file is searched into a per-file
/// buffer (no global lock during search), then the buffered stdout/stderr are merged into
/// the real `out` under a brief mutex and the aggregate `OperandResult` is maxed. In-file
/// order is preserved; cross-file order is the accepted worker-scheduling delta.
/// "Hidden" is the platform's own reading — a leading dot everywhere plus the hidden file
/// attribute on Windows — so a walk there can skip a file a unix one would search. A
/// redirected member ([`StdoutDest::File`]) gets a [`ChunkedOut`], so memory follows the
/// chunk rather than the whole result.
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
    // What one worker holds of a file: a whole file's matches (bounded by the member's own
    // cap when captured) or a fixed chunk.
    let dest = out.dest;

    let mut builder = ignore::WalkBuilder::new(&root_abs);
    builder.filter_entry({
        // The filter closure is `'static` (it owns its captures), so it gets its own copy of
        // the root label; the walker's closure keeps the original.
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

    // Shared output + aggregate result; the search runs outside the lock, which is held
    // only for the hand-off, and the worker closures capture the owned state by reference so
    // the per-thread builder closure can be invoked more than once.
    //
    // Accepted parallel-walk trade-offs for the destinations that keep the per-file buffer:
    // (1) above OUTPUT_CAP, which files' lines survive is non-deterministic — only the count
    // is preserved; (2) a *piped* walk is uncapped, so a worker holds one whole matching
    // file (the dominant `| head` case ends the walk early via SIGPIPE instead).
    let shared_out = std::sync::Mutex::new(&mut *out);
    let shared_result = std::sync::Mutex::new(OperandResult::NoMatch);
    let root_abs = &root_abs;
    let root_display = &root_display;
    let out_ref = &shared_out;
    let result_ref = &shared_result;

    // The per-entry rules both walk shapes share: skips, path spelling, error reporting.
    let visit = |entry: Result<ignore::DirEntry, ignore::Error>, sink: &mut dyn LineOut| {
        match entry {
            Ok(e) => {
                let ft = e.file_type();
                if ft.is_some_and(|t| t.is_symlink() || t.is_dir()) {
                    return OperandResult::NoMatch; // grep -r skips symlinks; dirs traversed by the walk
                }
                let display = traversal_display(root_display, root_abs, e.path(), SHELL_PLATFORM);
                if !file_allowed_by_filters(&display, spec, SHELL_PLATFORM) {
                    return OperandResult::NoMatch;
                }
                search_file(e.path(), &display, spec, matcher, sink, show_prefix)
            }
            Err(e) => {
                // Unreadable dir/file during traversal → grep-style error.
                let (path, message) = walk_error_info(&e);
                let display = path.as_deref().map_or_else(
                    || op.display.clone(),
                    |p| traversal_display(&op.display, root_abs, p, SHELL_PLATFORM),
                );
                emit_error(spec, sink, &display, &message);
                OperandResult::Error
            }
        }
    };

    if dest == StdoutDest::File {
        builder.build_parallel().run(|| {
            let mut local = ChunkedOut::new(out_ref);
            Box::new(move |entry| {
                let r = visit(entry, &mut local);
                local.hand_off(); // this entry's tail: nothing outlives its file
                let mut gr = result_ref.lock().unwrap_poison();
                *gr = (*gr).max(r);
                ignore::WalkState::Continue
            })
        });
    } else {
        builder.build_parallel().run(|| {
            let mut local = PerFileOut::new(dest);
            Box::new(move |entry| {
                let r = visit(entry, &mut local);
                let (buf, err, block_pending) = local.take_stdio();
                // An entry that wrote nothing (a directory, a symlink, a filtered file) never
                // reaches the lock. `take_stdio` still runs above: it resets the cap accounting.
                if !buf.is_empty() || !err.is_empty() {
                    merge_worker_output(
                        &mut out_ref.lock().unwrap_poison(),
                        &buf,
                        &err,
                        block_pending,
                    );
                }
                let mut gr = result_ref.lock().unwrap_poison();
                *gr = (*gr).max(r);
                ignore::WalkState::Continue
            })
        });
    }

    shared_result.into_inner().unwrap_poison()
}

/// Hand a walk worker's matches and stderr to the member's sink under its lock, then
/// flush: a piped tail sees the data as it lands. `block_pending` marks the hand-off
/// carrying the entry's first content record, where the separator the member may owe it is
/// written under that same lock — which is why a worker only records the decision.
fn merge_worker_output(guard: &mut Output, buf: &[u8], err: &[u8], block_pending: bool) {
    if block_pending {
        guard.file_block();
    }
    guard.write_bytes(buf);
    for chunk in err.chunks(4096) {
        guard.write_err(&String::from_utf8_lossy(chunk));
    }
    guard.flush();
}

/// Bytes of one entry's output a redirected walk's worker holds before handing them over.
const WALK_CHUNK: usize = 64 * 1024;

/// The sink of a redirected walk's worker: it accumulates one entry's records in a chunk
/// and hands them over once full — at a record end, or between the records of one
/// [`LineOut::flush`] — plus the entry's tail when it ends, so a worker holds [`WALK_CHUNK`]
/// rather than a whole file's matches. An entry exceeding the chunk therefore reaches the
/// sink as several runs other workers' runs may interleave: the cross-file ordering the
/// walk rows already relax, at chunk grain (in-file order is preserved).
struct ChunkedOut<'a> {
    shared: &'a std::sync::Mutex<&'a mut Output>,
    buf: Vec<u8>,
    err: Vec<u8>,
    block_pending: bool,
}

impl<'a> ChunkedOut<'a> {
    fn new(shared: &'a std::sync::Mutex<&'a mut Output>) -> Self {
        ChunkedOut {
            shared,
            buf: Vec::new(),
            err: Vec::new(),
            block_pending: false,
        }
    }

    /// Give everything this worker holds to the member's sink (one lock hold). Both callers
    /// are record boundaries, so only whole records go out: another worker can hand off
    /// between records, never inside one.
    fn hand_off(&mut self) {
        // An entry that wrote nothing hands nothing over and opens no block.
        if self.buf.is_empty() && self.err.is_empty() {
            return;
        }
        let block_pending = std::mem::take(&mut self.block_pending);
        merge_worker_output(
            &mut self.shared.lock().unwrap_poison(),
            &self.buf,
            &self.err,
            block_pending,
        );
        self.buf.clear();
        self.err.clear();
    }
}

impl LineOut for ChunkedOut<'_> {
    fn write_bytes(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }

    fn write_err(&mut self, s: &str) {
        self.err.extend_from_slice(s.as_bytes());
    }

    /// The Sink flushes at record ends only, so this is where a full chunk goes.
    fn flush(&mut self) {
        self.hand_off_if_full();
    }

    /// Hold at most [`WALK_CHUNK`]: a shape writing several records under one [`flush`]
    /// (the `-o` per-match loop) calls this between them.
    fn hand_off_if_full(&mut self) {
        if self.buf.len() >= WALK_CHUNK {
            self.hand_off();
        }
    }

    fn file_block(&mut self) {
        self.block_pending = true;
    }
}

/// A captured or piped walk's worker sink: one entry's per-file buffer, bounded by the
/// member's cap, recording the block opening instead of writing a separator. The pending
/// flag lives here so the member's writer carries only its own answered latch.
struct PerFileOut {
    out: Output,
    block_pending: bool,
}

impl PerFileOut {
    fn new(dest: StdoutDest) -> Self {
        PerFileOut {
            out: Output::new(OutputSink::Buffer(Vec::new()), dest),
            block_pending: false,
        }
    }

    /// The entry's stdout/stderr and whether those bytes open a block; the per-file cap
    /// accounting resets here.
    fn take_stdio(&mut self) -> (Vec<u8>, Vec<u8>, bool) {
        let block_pending = std::mem::take(&mut self.block_pending);
        let (buf, err) = self.out.take_stdio();
        (buf, err, block_pending)
    }
}

impl LineOut for PerFileOut {
    fn write_bytes(&mut self, b: &[u8]) {
        self.out.write_bytes(b);
    }

    fn write_err(&mut self, s: &str) {
        self.out.write_err(s);
    }

    fn flush(&mut self) {
        self.out.flush();
    }

    fn file_block(&mut self) {
        self.block_pending = true;
    }
}

/// Display path for a walked entry: the operand spelling plus relative suffix, joined in
/// the platform's separator so the displayed path is spelled the way the operand was.
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
        ShellPlatform::Unix => format!("{}/{}", root_display.trim_end_matches('/'), rel),
        ShellPlatform::Windows => windows::traversal_display(root_display, &rel),
    }
}

fn walk_error_info(e: &ignore::Error) -> (Option<PathBuf>, String) {
    if let ignore::Error::WithPath { path, err } = e {
        (Some(path.clone()), normalize_io_message(err))
    } else {
        (None, normalize_io_message(e))
    }
}

/// Strip the ` (os error N)` suffix so the phrasing matches BSD grep's.
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

/// Search one file with BSD binary semantics (32 KiB NUL detection; the `Binary file …
/// matches` message instead of lines unless `-a`; matches anywhere count for the exit code,
/// and matching is byte-oriented so invalid-UTF-8 match lines are not poisoned).
fn search_file(
    path: &Path,
    display: &str,
    spec: &EngineSpec,
    matcher: &grep_regex::RegexMatcher,
    out: &mut dyn LineOut,
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
    // Non-seekable operands (FIFOs, /dev/*) cannot be re-read: stream the buffered head and
    // then the remainder — BSD grep likewise never seeks.
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

fn emit_error(spec: &EngineSpec, out: &mut dyn LineOut, display: &str, message: &str) {
    if !spec.flags.s {
        // io errors carry a " (os error N)" suffix; BSD grep prints the bare message.
        out.write_err(&format!(
            "{}: {display}: {}\n",
            spec.verb,
            strip_os_error(message)
        ));
    }
}

/// The line the host command prints between the blocks of different files — and between the
/// groups of one file — in a context search.
const BLOCK_SEPARATOR: &[u8] = b"--\n";

#[expect(clippy::struct_excessive_bools)] // grep CLI flag surface
struct GrepSink<'a> {
    spec: &'a EngineSpec,
    display: &'a str,
    show_prefix: bool,
    binary: bool,
    message_emitted: bool,
    selected_any: bool,
    block_pending: bool,
    count: u64,
    matcher: &'a SearchMatcher,
    out: &'a mut dyn LineOut,
}

impl grep_searcher::Sink for GrepSink<'_> {
    type Error = io::Error;

    fn matched(
        &mut self,
        _searcher: &grep_searcher::Searcher,
        mat: &grep_searcher::SinkMatch<'_>,
    ) -> Result<bool, io::Error> {
        if self.spec.flags.count_mode() {
            // Count selected lines; -l stops at the first. Binary files get the plain
            // count/name, and byte-oriented matching still counts invalid-UTF-8 lines.
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
        self.open_block();
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
                self.out.write_bytes(b"\n");
                // One flush covers the line's whole match set, so a chunk filled inside it
                // is handed over here.
                self.out.hand_off_if_full();
            }
        } else {
            self.write_prefix(mat.line_number(), false);
            self.out.write_bytes(content);
            self.out.write_bytes(b"\n");
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
        self.open_block();
        let content = trim_line_terminator(ctx.bytes(), SHELL_PLATFORM);
        self.write_prefix(ctx.line_number(), true);
        self.out.write_bytes(content);
        self.out.write_bytes(b"\n");
        self.out.flush();
        Ok(true)
    }

    fn context_break(&mut self, _searcher: &grep_searcher::Searcher) -> Result<bool, io::Error> {
        self.out.write_bytes(BLOCK_SEPARATOR);
        Ok(true)
    }
}

impl GrepSink<'_> {
    /// The file's first match or context line: tell the sink a block starts here, once.
    fn open_block(&mut self) {
        if std::mem::take(&mut self.block_pending) && self.spec.flags.context_active() {
            self.out.file_block();
        }
    }

    /// Write the line prefix: `path` + separator + `lineno` + separator. Match lines use `:`
    /// (path separator replaced by NUL with --null); context lines use `-`.
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
            self.out.write_bytes(&[sep]);
        }
        if let Some(n) = lineno {
            self.out.write_bytes(n.to_string().as_bytes());
            self.out.write_bytes(if is_context { b"-" } else { b":" });
        }
    }

    /// Emit the -c count line and/or the -l name line after the scan, in BSD order (count
    /// first). Zero-count files always print their count; -l prints the name only with a
    /// selection, and --null NUL-terminates the name only.
    fn finish_count(&mut self) {
        if self.spec.flags.c {
            if self.show_prefix {
                self.out.write_bytes(self.display.as_bytes());
                self.out.write_bytes(b":");
            }
            self.out.write_bytes(self.count.to_string().as_bytes());
            self.out.write_bytes(b"\n");
        }
        if self.spec.flags.l && self.selected_any {
            self.out.write_bytes(self.display.as_bytes());
            self.out
                .write_bytes(if self.spec.flags.null { b"\0" } else { b"\n" });
        }
        self.out.flush();
    }
}

/// Drop a line's terminator. On Windows a CRLF line's `\r` goes with its `\n`, so a served
/// grep never leaks carriage returns into agent-visible lines; a lone trailing `\r` is
/// content, exactly as on unix — the pair is stripped as a pair.
fn trim_line_terminator(b: &[u8], platform: ShellPlatform) -> &[u8] {
    match platform {
        ShellPlatform::Unix => b.strip_suffix(b"\n").unwrap_or(b),
        ShellPlatform::Windows => b
            .strip_suffix(b"\r\n")
            .or_else(|| b.strip_suffix(b"\n"))
            .unwrap_or(b),
    }
}

/// Streams matched lines as they are produced (an early-exit tail stops the search as it
/// does for the real grep). A captured member self-caps at [`OUTPUT_CAP`] bytes, after
/// which writing stops but the search continues (exit codes stay correct for huge outputs);
/// an unbounded member (piped or redirected) streams the whole result ([`StdoutDest`] owns
/// the reading). A write to a closed pipe kills the process: SIGPIPE on unix, the
/// `BrokenPipe` error on Windows ([`exit_on_broken_pipe`]). Stderr is buffered.
struct Output {
    sink: OutputSink,
    err: Vec<u8>,
    written: usize,
    dest: StdoutDest,
    /// Whether a block of lines has been printed: the next file's first block is separated
    /// from it by [`BLOCK_SEPARATOR`].
    block_printed: bool,
}

enum OutputSink {
    Stdout(io::BufWriter<io::Stdout>),
    // In-memory sink: the parallel-walk per-worker buffers and every in-process pin.
    Buffer(Vec<u8>),
}

/// The write side of a served search: the member's own sink ([`Output`]) or a walk
/// worker's ([`ChunkedOut`]). Where the bytes end up is the sink's business.
trait LineOut {
    fn write_bytes(&mut self, b: &[u8]);
    fn write_err(&mut self, s: &str);
    /// Show what is written to the consumer, per record, so a closed pipe stops the search.
    fn flush(&mut self);
    /// Hand over what this sink holds once a chunk accumulated — only the walk's chunk sink
    /// holds anything — and the hand-off point *inside* a run of records that one [`flush`]
    /// covers (`-o` writes a record per match), so a chunk filled there is not held to the
    /// run's end.
    fn hand_off_if_full(&mut self) {}

    /// The file's first printed block starts here — its match or context lines, never the
    /// binary notice. With context active the host opens a further file's block with
    /// [`BLOCK_SEPARATOR`]: the member's sink writes the separator before the block's own
    /// bytes, a walk worker's sink only records it for [`merge_worker_output`].
    fn file_block(&mut self);
}

impl LineOut for Output {
    fn write_bytes(&mut self, b: &[u8]) {
        let take = match self.dest.limit() {
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

    /// Error message (stderr) — bounded.
    fn write_err(&mut self, s: &str) {
        if self.err.len() < 64 * 1024 {
            self.err.extend_from_slice(s.as_bytes());
        }
    }

    /// Push buffered stdout out (per line) so `head` sees data and its early exit reaches us
    /// through [`exit_on_broken_pipe`]. A file destination needs no pacing and rides the
    /// buffer, so a huge redirected result pays no syscall per line.
    fn flush(&mut self) {
        if self.dest == StdoutDest::File {
            return;
        }
        if let OutputSink::Stdout(w) = &mut self.sink {
            exit_on_broken_pipe(&w.flush());
        }
    }

    fn file_block(&mut self) {
        if std::mem::replace(&mut self.block_printed, true) {
            self.write_bytes(BLOCK_SEPARATOR);
        }
    }
}

/// The closed-pipe exit shared by both stdout writers: on unix SIGPIPE already ended the
/// process, so the write error is dropped; on Windows a `BrokenPipe` error is what a closed
/// pipe surfaces as, and it kills the process with [`BROKEN_PIPE_EXIT`].
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
    fn new(sink: OutputSink, dest: StdoutDest) -> Self {
        Output {
            sink,
            err: Vec::new(),
            written: 0,
            dest,
            block_printed: false,
        }
    }

    /// Drain the buffered stdout and stderr. For the buffer sink the stdout Vec is taken;
    /// for the stdout sink nothing is buffered. Resets `written` (the per-file cap accounting).
    fn take_stdio(&mut self) -> (Vec<u8>, Vec<u8>) {
        self.written = 0;
        match &mut self.sink {
            OutputSink::Buffer(v) => (std::mem::take(v), std::mem::take(&mut self.err)),
            OutputSink::Stdout(_) => (Vec::new(), std::mem::take(&mut self.err)),
        }
    }

    /// Flush stdout and emit the buffered stderr, through the same closed-pipe rule as every
    /// other write (the search is over, so a break here cannot extend the walk).
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

#[cfg(all(test, target_os = "macos"))]
mod parity_tests {
    use super::test_support::{engine_run, engine_run_with_stdin, sorted_lines};
    use super::*;
    use std::process::Command;

    /// Build the fixture tree used by every matrix row.
    fn build_fixture(ws: &Path, home: &Path) {
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
        // Exclusion-delta tree: hidden + gitignored content (a .git dir is needed for gitignore).
        fs::create_dir_all(ws.join("ign/.git")).unwrap();
        fs::write(ws.join("ign/visible.txt"), "x1\n").unwrap();
        fs::write(ws.join("ign/.hidden.txt"), "x2\n").unwrap();
        fs::write(ws.join("ign/.gitignore"), "*.log\n").unwrap();
        fs::write(ws.join("ign/skip.log"), "x3\n").unwrap();
        fs::create_dir_all(ws.join("bindir")).unwrap();
        fs::write(ws.join("bindir/bin1.dat"), b"hello\x00world\nneedle\n").unwrap();
        fs::write(ws.join("bindir/bin2.dat"), b"needle\x00world\n").unwrap();
        fs::write(ws.join("bindir/bin3.dat"), b"hello\x00world\n").unwrap();
        fs::write(ws.join("bindir/bin4.dat"), b"needle\nhello\x00world\n").unwrap();
        // NUL in the window plus an invalid-UTF-8 match line: byte-oriented matching counts
        // the match (message/exit 0, raw line under -a) instead of poisoning it.
        fs::write(ws.join("bindir/bin5.dat"), b"\xff\x00needle\n").unwrap();
        // Invalid-UTF-8 line, no match: silent exit 1.
        fs::write(ws.join("bindir/bin6.dat"), b"\xff\x00other\n").unwrap();
        std::os::unix::fs::symlink("a.txt", ws.join("filelink")).unwrap();
        std::os::unix::fs::symlink("plain", ws.join("dirlink")).unwrap();
        std::os::unix::fs::symlink("plain", ws.join("plain/dirlink2")).unwrap();
        std::os::unix::fs::symlink("missing", ws.join("broken")).unwrap();
        fs::create_dir_all(home.join("htree/sub")).unwrap();
        fs::write(home.join("htree/f1.txt"), "needle\n").unwrap();
        fs::write(home.join("htree/sub/f2.txt"), "needle\n").unwrap();
        fs::create_dir_all(ws.join("sub")).unwrap();
        fs::write(ws.join("sub/s.txt"), "needle\n").unwrap();
        // Context rows: two entries whose `-C1` output exceeds the redirected walk's hand-off
        // chunk (so the chunked destination really interleaves), plus a small entry.
        fs::create_dir_all(ws.join("walkctx")).unwrap();
        let walk_block = "needle\nfiller\nfiller\nfiller\nfiller\n".repeat(1000);
        fs::write(ws.join("walkctx/one.txt"), &walk_block).unwrap();
        fs::write(ws.join("walkctx/two.txt"), &walk_block).unwrap();
        fs::write(ws.join("walkctx/small.txt"), "x\nneedle\nx\n").unwrap();
        // Literal filename operands for quoted/escaped glob and ~ forms: BSD grep searches
        // these literally when the shell quotes/escapes the metacharacters.
        fs::write(ws.join("~"), "needle\n").unwrap();
        fs::write(ws.join("*.txt"), "needle\n").unwrap();
        fs::write(ws.join("a\"b.txt"), "needle\n").unwrap();
        // Quoted `~` + unquoted glob: a literal `~` dir relative to cwd.
        fs::create_dir_all(ws.join("sub/~")).unwrap();
        fs::write(ws.join("sub/~/q.txt"), "needle\n").unwrap();
    }

    /// Fresh temp root with the fixture `ws` and `home` trees; bind the TempDir for the test.
    fn fixture() -> (tempfile::TempDir, PathBuf, PathBuf) {
        super::test_support::open_the_gate();
        let tmp = tempfile::tempdir().expect("tempdir");
        let ws = tmp.path().join("ws");
        let home = tmp.path().join("home");
        fs::create_dir_all(&ws).expect("ws");
        fs::create_dir_all(&home).expect("home");
        build_fixture(&ws, &home);
        (tmp, ws, home)
    }

    /// Run the ORIGINAL command through a real shell — the authentic reference (globs, `cd`,
    /// pipes and BSD grep all apply).
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

    /// Run the real grep with the spec's fallback argv (the grep member only: pipeline rows).
    fn real_run_fallback(spec: &EngineSpec) -> (Vec<u8>, Vec<u8>, i32) {
        real_run_fallback_with_stdin(spec, &[])
    }

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
        // Real grep may exit early (-m); a write error (EPIPE) is then the producer's native
        // death, not a test failure. Dropping the handle delivers EOF for the full-scan case.
        let _ = stdin_pipe.write_all(stdin);
        drop(stdin_pipe);
        let out = child.wait_with_output().expect("system grep runs");
        (
            out.stdout,
            out.stderr,
            out.status.code().expect("grep exits normally"),
        )
    }

    /// The full command prefix before the served grep member — pipeline members plus any
    /// preceding `&&`/`||`/`;`-joined segments (all preserved verbatim). stdin-fed rows only.
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
        // Walk back over any preceding segment (pipes, `&&`, `||`, `;`, blank lines): the
        // rewrite preserves them verbatim, so the prefix assertion needs them. Rows with
        // non-pipe segments before the producer must have patterns that don't match those
        // segments' own output, which the authentic pipeline's grep never sees.
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

    /// Rejoin `(segment, connector)` pairs from `start` on, prefixed by that segment's own
    /// connector — the verbatim-tail shape both tail helpers share.
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

    /// The original command's text from its first `|`/`|&` connector on — the tail that must
    /// survive the rewrite verbatim.
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

    /// Analyze and join as the production caller does for this parity lane's platform, so the
    /// assertions read one command's rewrite. The executable path it names is a stand-in.
    fn analyze_joined(
        command: &str,
        ws: &Path,
        home: &Path,
        allow_single: bool,
    ) -> Result<JoinedAnalyzeOutput, AnalyzeFailure> {
        let analyzed = analyze_command(command, ws, home, ShellPlatform::Unix, allow_single)?;
        let jsons: Vec<String> = analyzed.specs.iter().map(spec_json).collect();
        let joined = join_rewritten(
            &analyzed.segments,
            &analyzed.specs,
            &jsons,
            ShellPlatform::Unix,
            "/mahbot",
            &analyzed.root_cwd,
        )
        .expect("the unix join never refuses");
        Ok((
            analyzed.specs,
            analyzed.shapes,
            joined.text,
            analyzed.outcomes,
        ))
    }

    /// Assert the engine's output/exit are byte-identical to the real grep. Recursive DIRECTORY
    /// walks may order files differently across workers, so those compare as sorted line-sets.
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

    /// The members after the served grep, rejoined — must survive the rewrite verbatim.
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

    /// stdin-fed parity: capture the producer's stream through a real shell, then feed the
    /// same bytes to the engine and the real grep.
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
        // Pins the stream-size marker chain. The count is the searcher's bytes through the
        // CountingReader — the head is re-read through it, so an EOF read reports the exact
        // stream length; only -m/-l early stops report a bounded prefix.
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

    /// A member whose output `sh` sends to a file: the engine's own stdout, handed the same
    /// redirect, must be byte-identical to what the ORIGINAL shell left in `target`, and the
    /// other stream must reach both captures identically. Output redirects only (`>`, `>>`,
    /// `&>`; a `2>` row is pinned by `unix_operator_pins`) and stderr-free rows. Reproduces
    /// the shell's write only below [`OUTPUT_CAP`]: the harness can never select
    /// [`StdoutDest::File`], so an oversized row would compare a capped stream to the file.
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
        assert!(
            eout.len() < OUTPUT_CAP && written.len() < OUTPUT_CAP,
            "{command}: a result at or above OUTPUT_CAP is not pinned by this capped \
             in-process harness"
        );
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

    fn assert_falls_back_reason(command: &str, ws: &Path, home: &Path, reason: &str) {
        let err = analyze_joined(command, ws, home, false)
            .err()
            .unwrap_or_else(|| panic!("{command}: expected fallback, got served"));
        assert_eq!(err.reason.to_string(), reason, "{command}");
    }

    #[expect(clippy::too_many_lines)] // differential parity matrix
    #[test]
    fn differential_parity_matrix() {
        let (_tmp, ws, home) = fixture();
        let rows: &[&str] = &[
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
            "egrep -n 'foo|bar' br.txt",
            "egrep -n '(foo)+' br.txt",
            "egrep -n 'a{2}' br2.txt",
            "fgrep -n 'a.b' br2.txt",
            "fgrep -n 'foo' a.txt",
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
            "grep -A1 -B1 foo a.txt b.txt",
            "grep -n -A1 b a.txt",
            "grep -C1 foo a.txt b.txt", // both operands print: one separator, byte for byte
            "grep -h -C1 foo a.txt b.txt", // no path prefix: the separator is still printed
            // an in-file break (`--` between the groups of one file) and a cross-file one together
            "grep -n -A1 -B1 'b\\|h' ctx.txt a.txt",
            "grep -C0 foo a.txt b.txt", // zero context: no separator
            // counting and listing modes print none (contexts accepted but ignored); binary
            // notices are neither printed around nor counted as printed output
            "grep -C1 needle sub/s.txt bindir/bin1.dat bindir/bin4.dat", // notices after a block: none at all
            "grep -C1 needle bindir/bin1.dat sub/s.txt", // a notice first: none before the block
            "grep -C1 needle sub/s.txt bindir/bin1.dat walkctx/small.txt", // notice between: one, before the last block
            "grep -rn -C1 x plain",
            "grep -rn -C1 'x1\\|x3' plain",
            "grep -rn -C1 needle walkctx",
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
            "grep -w foo a.txt",
            "grep -w café uni.txt",
            "grep -x foo x1.txt",
            "grep -x '^foo' x1.txt",
            "grep -x '' z2.txt",
            "grep -x -e foo -e bar a.txt",
            "grep -iw Foo a.txt",
            "grep -m2 a m.txt",
            "grep -m2 -v a m.txt",
            "grep -m2 a m.txt c.txt",
            "grep -m1 -A1 a m.txt",
            "grep -m1 -B1 b m.txt",
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
            "grep -l --null foo a.txt b.txt",
            "grep -c --null foo a.txt b.txt",
            "grep -cl --null foo a.txt b.txt",
            "grep -lr --null x plain",
            "grep -cr --null x plain",
            "grep -clr --null x plain",
            "grep x missing.txt a.txt",
            "grep -s x missing.txt a.txt",
            "grep x plain a.txt",
            "grep x xx.txt/ a.txt",
            "grep -rn x plain missing.txt",
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
            // Binary + invalid-UTF-8 match line: byte-oriented detection must count the match
            // and -a must print the raw line — the per-line UTF-8 gate must not poison it.
            "grep -rn needle bindir/bin5.dat bindir/bin4.dat",
            "grep -a needle bindir/bin5.dat bindir/bin4.dat",
            "grep -rn needle bindir/bin5.dat bindir/bin6.dat",
            "grep -r x plain/dirlink",
            "grep -r x plain/dirlink/",
            "grep -r foo filelink",
            "grep foo filelink a.txt",
            "grep -r x plain/dirlink2",
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
            "cd sub && grep -r needle .",
            "cd sub && grep -rn needle .",
            "grep -rn needle ~/htree",
            "grep -rn needle ~/htree/sub",
            "grep -rn needle sub",
            "grep -rn needle plain/*.txt",
            "grep -rn needle plain/d1/*.txt",
            "grep -rn needle ~/htree/*.txt",
            "grep -n -- x a.txt",
            "grep x -n a.txt",
            "grep -e x a.txt b.txt",
            "grep -e x -e y a.txt b.txt",
            "grep -y x a.txt",
            "grep -u -n x a.txt",
            "grep -r x -n plain",
            "grep -rn -- -x dash.txt",
            "grep -e x -- -x1", // dash-prefixed operand after `--`
            "grep -rn x plain | head -5",
            "grep -rn x plain | head",
            // Producer-first pipelines: the first grep is served from the producer's stdin
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
            "grep -rn needle '~'",
            "grep -rn needle '*.txt'",
            "grep -rn needle \\*.txt",
            "grep -rn needle 'a\"b.txt'",
            "grep -rn needle '~/htree'", // literal path, no such file (stderr row)
            "grep -rn needle '~user'",   // literal path, no such file (stderr row)
            "grep -rn needle ~/\"htree\"", // mixed form: `~/` expands, quoted rest joins
            // any-length pipelines: any tail is served verbatim, with grep member parity (tails
            // are never analyzed: second greps, grep introducers, 3+ members). Under 2>&1
            // engine stderr is buffered to finish while BSD emits at operand-open, so
            // missing-first or order-sensitive tails diverge — accepted residual
            "grep -rn x plain | wc -l",
            "grep -rn x plain | tail -2",
            "grep -rn x plain | sort",
            "grep -rn x plain missing.txt 2>&1 | wc -l",
            "grep -rn x plain | grep -v 'plain/d1' | head -5", // 3+ member, second grep preserved
            "grep foo a.txt | grep oo", // 2-member grep|grep (tail grep preserved)
            "grep foo a.txt | xargs grep oo", // grep introducer preserved in the tail
            "grep foo a.txt | head -1 | wc -l", // 3+ member, single-file first grep
            // Quoted `~` + unquoted glob is a literal cwd-relative path (no home-strip).
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

    /// The glued spellings, observed end-to-end: the rewrite's own stream lands in the file
    /// the command glued the operator to, and a backgrounded search keeps what follows its `&`.
    #[test]
    fn glued_operator_and_background_parity() {
        let (_tmp, ws, home) = fixture();

        // A glued target (`>file`, `>>file`, `&>file`), the spaced control, and the recursive one.
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

        // A backgrounded search: served, with everything after its `&` left to the shell.
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

        // A comment takes the rest of its line with it: the served search is the separated
        // spelling's word set, and the original's `sh` runs nothing after the `#`.
        assert_parity("grep -rn x plain # note & echo TAIL-RAN", &ws, &home);
    }

    #[test]
    fn stdin_m_early_stop_stops_consuming() {
        // `seq 1 100000 | grep -m1 5`: the match sits inside the 32 KiB head, so a -m1 serve
        // must stop reading once found. The marker reports consumed bytes, well short of the
        // full stream.
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
        // Pinned reason labels: `;;` is a case terminator (compound, not an empty member);
        // empty pipeline members stay rejected; non-grep pipelines are "no grep".
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
        // Any-length pipelines with a single-file first grep fall back for the perf gate, not
        // the pipeline shape.
        assert_falls_back_reason("grep x a.txt | head -1 | wc -l", &ws, &home, "single file");
        assert_falls_back_reason("grep x a.txt | grep y", &ws, &home, "single file");
        assert_falls_back_reason("grep x a.txt | xargs grep y", &ws, &home, "single file");
    }

    #[test]
    fn per_segment_skip_with_served_sibling() {
        // A single-file grep (perf gate) does not poison a sibling servable recursive grep:
        // the unservable member is kept verbatim and the whole command is served.
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

        // A compound (`if`) containing a grep is skipped while a sibling recursive grep is served.
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
        // Escape-aware substitution scans change segmentation: an escaped backtick no longer
        // truncates a backtick span, so the trailing grep is a real member and gets served
        // (the old scan swallowed it into an unterminated span → fallback). The
        // serve→fallback direction is pinned in `fallback_triggers`.
        let (_tmp, ws, home) = fixture();
        analyze_joined("echo `a\\`b` ; grep x a.txt b.txt", &ws, &home, true)
            .expect("escaped backtick: trailing grep is a separate served member");
    }

    #[test]
    fn exclusion_delta_is_rg_default() {
        // An approved delta: recursive walks skip hidden/gitignored content; explicit operands
        // are always searched.
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
        // The engine must never silently broaden/narrow a match set: untranslatable or
        // uncompilable patterns fall back instead.
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

// The token classifier is shared with the read-only guard (readonly.rs); these rows pin
// the TokenKind→(redirect, needs_target) mapping for the divergent redirect shapes.

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
        // `10>` (multi-digit fd) expects a target word; `1>0` does not (next word is an operand).
        let (redirects, operands) = redirects_of("grep foo a.txt 10> out.txt");
        assert_eq!(redirects, ["10>", "out.txt"], "10> must consume its target");
        assert_eq!(operands, ["a.txt"]);

        let (redirects, operands) = redirects_of("grep foo a.txt 1>0 out.txt");
        assert_eq!(redirects, ["1>0"], "1>0 is self-contained");
        assert_eq!(operands, ["a.txt", "out.txt"], "out.txt stays an operand");
    }
}

// The cd option-skip/target-extract loop is shared with the read-only guard (readonly.rs);
// these rows pin the three-state scan and the engine's fail-closed mapping. The guard's own
// fail-closed arm is pinned by `cd_flag_forms_fail_closed` and `cd_extra_operands_fail_closed`.

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
        // (words after the verb, expected scan); target rows carry the stripped target.
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

        // The engine's fail-closed mapping: bare → $HOME; bad option / operand → fallback.
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

// The Windows matcher is pure Rust, so the host's own `fnmatch` is what it is pinned
// against; both have the semantics the unix lane uses (filters and glob expansion), and the
// cross-product below — malformed spellings included — is this matcher's whole semantics pin.

#[cfg(all(test, unix))]
mod windows_fnmatch_parity {
    use super::*;

    /// This host's `libc::fnmatch`, called exactly as [`fnmatch_flags`] calls it on unix.
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
            // Malformed classes: an unterminated bracket and a reversed range.
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

// Both splitters run the shared core (shell::segment_command); these rows pin the two
// policies' deliberate divergence so a silent swap fails loudly (unifying them is separate).

#[cfg(test)]
mod segmenter_pins {
    use super::super::extract_command_segments;
    use super::*;

    #[test]
    fn divergent_policies_stay_pinned() {
        // (input, profile segments) — profile skips empty segments and drops a backslash before
        // ordinary chars.
        let profile_rows: &[(&str, &[&str])] = &[
            // Unquoted backslash before an ordinary char: dropped (kept only before escapes).
            ("echo \\a", &["echo a"]),
            // `|&` is not a compound connector: the `&` starts its own segment.
            ("a |& b", &["a", "& b"]),
            // `;;` outside a case: two `;` separators, empty segment skipped.
            ("echo a ;; echo b", &["echo a", "echo b"]),
            // `sh`'s comment is ordinary text to the profile splitter, which never rewrites.
            ("grep x a.txt # c & echo A", &["grep x a.txt # c & echo A"]),
        ];
        for (input, expected) in profile_rows {
            let segs = extract_command_segments(input);
            let got: Vec<&str> = segs.iter().map(String::as_str).collect();
            assert_eq!(got, *expected, "profile: {input:?}");
        }

        // (input, grep (segment, connector) pairs) — grep errors on empty members (except blank
        // lines and `;;` case-arm terminators) and always preserves backslashes.
        let grep_ok: &[(&str, &[(&str, &str)])] = &[
            // Backslash always preserved: dropping `\*` would turn the operand into a glob.
            ("grep \\*.txt a.txt", &[("grep \\*.txt a.txt", "")]),
            // `|&` is a single compound connector (stderr merge).
            ("a |& grep x a.txt", &[("a", "|&"), ("grep x a.txt", "")]),
            // An unquoted `#` comments out the rest of its line only: the `&` inside it and the
            // member after it survive as neither connector nor command.
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

// The platform split in these helpers keeps a unix answer unix-shaped: `\` is an ordinary
// filename character there, and a walked entry is always `/`-joined.

#[cfg(test)]
mod platform_string_pins {
    use super::*;

    #[test]
    fn unix_display_paths_and_filters_read_only_slash() {
        // A walked entry joins on `/`, including the degenerate empty operand spelling.
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

    /// A `\r` is dropped only as the `\n`'s own carriage return: a line whose last byte is a
    /// bare `\r` (no final newline) keeps it on both platforms, as the host grep delivers it.
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

    /// One canonicalization for both sides of the cwd gate: on Windows the verbatim (`\\?\`)
    /// prefix is stripped so tracked and displayed paths stay naturally spelled, and a path
    /// that cannot be canonicalized falls back to its lexical spelling.
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

// Both hand-offs are driven from this host: the spec rides argv on unix and a scratch file
// on Windows, and the Windows executable quoting is the one part depending on install path.

#[cfg(test)]
mod handoff_pins {
    use super::super::SpecFiles;
    use super::*;

    /// What the hand-off makes of one executable spelling.
    enum HandOff {
        /// A fragment quoting the executable in place, backed by one scratch spec file.
        Served,
        /// A refusal naming what cmd.exe would rewrite inside the quotes.
        Refused,
    }

    struct HandOffCase {
        name: &'static str,
        exe: &'static str,
        verdict: HandOff,
    }

    /// The scratch-file transport's read half: what [`read_spec`] makes of one argv.
    enum SpecRead {
        Reads(&'static str),
        Unreadable,
    }

    /// The Windows hand-off, both halves as data: the executable quoting (and what cmd.exe
    /// rewrites even inside quotes) and the scratch file's reading. An installation under
    /// `C:\Program Files (x86)\…` is served, not disabled machine-wide.
    #[test]
    #[expect(clippy::too_many_lines)] // two cohesive tables over the hand-off
    fn the_windows_hand_off_round_trips() {
        let spec = render_only_json();
        let cases: &[HandOffCase] = &[
            HandOffCase {
                name: "a program files path is quoted in place",
                exe: r"C:\Program Files (x86)\MahBot\mahbot.exe",
                verdict: HandOff::Served,
            },
            HandOffCase {
                name: "a percent in the path is rewritten by cmd",
                exe: r"C:\Users\a%b\mahbot.exe",
                verdict: HandOff::Refused,
            },
            HandOffCase {
                name: "a bang in the path is rewritten by cmd",
                exe: r"C:\Users\a!b\mahbot.exe",
                verdict: HandOff::Refused,
            },
        ];
        for case in cases {
            let mut files = Vec::new();
            match case.verdict {
                HandOff::Served => {
                    let ServedMember { text, argv } =
                        render_served(&spec, &[], ShellPlatform::Windows, case.exe, &mut files)
                            .expect("a programme path cmd.exe reads literally inside quotes");
                    // cmd.exe keeps the quoting's content literal: the quoted exe is the prefix.
                    assert!(
                        text.starts_with(&format!("\"{}\" ", case.exe)),
                        "{}: {text}",
                        case.name
                    );
                    assert!(
                        text.contains(windows::SPEC_FILE_FLAG),
                        "{}: {text}",
                        case.name
                    );
                    // The same member's argv names the same words, with the spec's path bare.
                    assert_eq!(
                        argv.as_deref(),
                        Some(
                            [
                                ENGINE_VERB.to_string(),
                                windows::SPEC_FILE_FLAG.to_string(),
                                files[0].to_string_lossy().into_owned(),
                            ]
                            .as_slice()
                        ),
                        "{}: the plan's argv names the scratch file the fragment does",
                        case.name
                    );
                    assert_eq!(
                        files.len(),
                        1,
                        "{}: the spec rides a scratch file",
                        case.name
                    );
                    assert!(files[0].exists(), "{}: {:?}", case.name, files[0]);
                    let file = files[0].clone();
                    drop(SpecFiles(files));
                    assert!(
                        !file.exists(),
                        "{}: the call's guard removes it: {file:?}",
                        case.name
                    );
                }
                HandOff::Refused => {
                    let err =
                        render_served(&spec, &[], ShellPlatform::Windows, case.exe, &mut files)
                            .err()
                            .expect("a path cmd.exe rewrites even inside quotes");
                    assert!(
                        err.to_string().contains("hand-off refused"),
                        "{}: {err}",
                        case.name
                    );
                    assert!(files.is_empty(), "{}: nothing is written", case.name);
                }
            }
        }

        // The scratch-file transport's read half: the flag arm takes the file's contents (the
        // flag is never parsed as JSON), an unreadable path errors, and the argv arm passes
        // a bare spec through.
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
        let cases: Vec<(&str, Vec<String>, SpecRead)> = vec![
            (
                "the flag reads the scratch file",
                file_args(&path),
                SpecRead::Reads(r#"{"version":1}"#),
            ),
            (
                "an unreadable spec file is an error",
                file_args(&tmp.path().join("no-such-spec.json")),
                SpecRead::Unreadable,
            ),
            (
                "the argv arm passes a bare spec through",
                vec![r#"{"version":2}"#.to_string()],
                SpecRead::Reads(r#"{"version":2}"#),
            ),
        ];
        for (name, args, expected) in cases {
            match expected {
                SpecRead::Reads(expected) => assert_eq!(
                    read_spec(&args).expect("the spec is read"),
                    expected,
                    "{name}"
                ),
                SpecRead::Unreadable => assert!(
                    read_spec(&args).is_err(),
                    "{name}: never a parse of the flag"
                ),
            }
        }
    }
}

// The writer branches on the destination's value, never on the host.

#[cfg(test)]
mod writer_pins {
    use super::*;

    /// The member's buffered stdout as it stands — the merge target of every hand-off.
    fn member_stdout(shared: &std::sync::Mutex<&mut Output>) -> Vec<u8> {
        let guard = shared.lock().unwrap_poison();
        let OutputSink::Buffer(bytes) = &guard.sink else {
            unreachable!("buffered sink")
        };
        bytes.clone()
    }

    /// The redirected walk's hand-off points (the one path where the engine writes a result
    /// it cannot hold): a full chunk goes at a record end, a run under one flush (`-o`) hands
    /// over between records, and the entry end hands over the rest — nothing outlives its file,
    /// nothing is dropped, and no half-record is handed over. No block is opened here, so the
    /// merge adds no separator.
    #[test]
    fn chunked_walk_sink_hands_off_whole_records_at_record_and_entry_ends() {
        let mut member = Output::new(OutputSink::Buffer(Vec::new()), StdoutDest::File);
        let shared = std::sync::Mutex::new(&mut member);
        let mut out = ChunkedOut::new(&shared);
        let line = b"bigdir/file.txt:12345:needle\n";

        // Sub-chunk holds ride the entry end: a newline-ended zero count, then a NUL-ended name.
        out.write_bytes(b"path:0\n");
        out.write_bytes(b"path\0");
        out.flush();
        assert!(
            member_stdout(&shared).is_empty(),
            "a sub-chunk buffer waits for the entry end, not a record end"
        );
        out.hand_off();
        assert_eq!(member_stdout(&shared), b"path:0\npath\0");

        // Above the chunk: handed over at the following record end, keeping the worker bounded.
        let mut expected = member_stdout(&shared);
        for _ in 0..(2 * WALK_CHUNK / line.len()) {
            out.write_bytes(line);
            out.flush();
            assert!(
                out.buf.len() < WALK_CHUNK + line.len(),
                "a worker hands off as it fills instead of holding the result"
            );
            expected.extend_from_slice(line);
        }
        out.hand_off();
        assert!(out.buf.is_empty());
        assert_eq!(member_stdout(&shared), expected);

        // A chunk filled mid-record waits: no flush happens inside a record.
        let before = expected.len();
        let filler = vec![b'x'; WALK_CHUNK + 1];
        let tail = b":needle\n";
        out.write_bytes(&filler);
        assert_eq!(
            member_stdout(&shared).len(),
            before,
            "a chunk filled mid-record waits for the record's end"
        );
        out.write_bytes(tail);
        out.flush();
        assert!(out.buf.is_empty());
        assert_eq!(
            member_stdout(&shared).len(),
            before + filler.len() + tail.len()
        );

        // A run of records under one flush (`-o`) hands over as it fills.
        let start = member_stdout(&shared).len();
        let group = 8 * WALK_CHUNK / line.len();
        for _ in 0..group {
            out.write_bytes(line);
            out.hand_off_if_full();
            assert!(
                out.buf.len() < WALK_CHUNK + line.len(),
                "a hand-off inside a record run keeps the worker at the chunk"
            );
        }
        assert_eq!(
            member_stdout(&shared).len() + out.buf.len(),
            start + group * line.len(),
            "a run's records are handed over as they fill, none lost"
        );
        out.hand_off();
        assert_eq!(member_stdout(&shared).len(), start + group * line.len());
    }

    /// The between-files separator is decided at the merge point, under the member's lock: the
    /// opening hand-off asks for it, so the member's first block is never preceded by one, a
    /// further file's block always is, and that file's later chunks add none.
    #[test]
    fn the_merge_point_owns_the_between_files_separator() {
        let mut member = Output::new(OutputSink::Buffer(Vec::new()), StdoutDest::File);
        let shared = std::sync::Mutex::new(&mut member);
        let mut out = ChunkedOut::new(&shared);

        // The member's first block: nothing precedes it.
        out.file_block();
        out.write_bytes(b"one:needle\n");
        out.hand_off();
        assert_eq!(member_stdout(&shared), b"one:needle\n");

        // A further file's opening: the separator goes before that block's bytes.
        out.file_block();
        out.write_bytes(b"two:needle\n");
        out.hand_off();
        assert_eq!(member_stdout(&shared), b"one:needle\n--\ntwo:needle\n");

        // The rest of the same file's chunks: its block is already open.
        out.write_bytes(b"two:more\n");
        out.hand_off();
        assert_eq!(
            member_stdout(&shared),
            b"one:needle\n--\ntwo:needle\ntwo:more\n"
        );

        // An entry that writes only stderr opens no block: no separator, and the
        // member's stdout is untouched.
        out.write_err("grep: gone.txt: No such file or directory\n");
        out.hand_off();
        assert_eq!(
            member_stdout(&shared),
            b"one:needle\n--\ntwo:needle\ntwo:more\n"
        );
    }

    /// The destination decides what the result is worth — not the search. A captured member
    /// stops at [`OUTPUT_CAP`]; a file the command redirects into gets the whole stream.
    #[test]
    fn the_destination_decides_whether_the_result_is_cut() {
        // ~380 KiB of short match lines: long enough to cross the cap, short enough that the
        // cut lands mid-line either way.
        const LINES: usize = 4000;
        let (_tmp, ws, _home) = serve_fixture();
        let line = "needle ".to_string() + &"x".repeat(90) + "\n";
        let huge = ws.join("huge.txt");
        fs::write(&huge, line.repeat(LINES)).expect("huge fixture");
        let other = ws.join("nomatch.txt");
        fs::write(&other, "nothing here\n").expect("second operand");

        // The spec is written rather than analyzed: this pin is about the writer, and `-h`
        // keeps the rendered length equal to the matching lines.
        let operand = |name: &str, path: &Path| Operand {
            display: name.to_string(),
            resolved: path.to_string_lossy().into_owned(),
            trailing_slash: false,
        };
        let spec = EngineSpec {
            version: PROTOCOL_VERSION,
            verb: "grep".into(),
            mode: MatchMode::Basic,
            flags: GrepFlags {
                h: true,
                ..GrepFlags::default()
            },
            filters: Vec::new(),
            exclude_dir: Vec::new(),
            patterns: vec!["needle".into()],
            operands: vec![operand("huge.txt", &huge), operand("nomatch.txt", &other)],
            cwd: ws.to_string_lossy().into_owned(),
            fallback: Vec::new(),
            piped: false,
            stdin: false,
            report_stream_bytes: false,
        };
        let as_pipeline_member = EngineSpec {
            piped: true,
            ..spec.clone()
        };
        let matcher = build_matcher(&spec.patterns, spec.mode, &spec.flags).expect("matcher");
        let run = |dest: StdoutDest| {
            let mut out = Output::new(OutputSink::Buffer(Vec::new()), dest);
            let consumed = std::cell::Cell::new(false);
            let code = serve_into(&spec, &matcher, &mut out, io::Cursor::new(&[]), &consumed);
            let Output {
                sink: OutputSink::Buffer(bytes),
                ..
            } = out
            else {
                unreachable!("buffered sink")
            };
            (bytes, code)
        };

        // What fd 1 is, plus what the member's stdout feeds (only the capture pipe is capped).
        assert_eq!(StdoutDest::of(&spec, true), StdoutDest::Capture);
        assert_eq!(StdoutDest::of(&as_pipeline_member, true), StdoutDest::Tail);
        assert_eq!(StdoutDest::of(&spec, false), StdoutDest::File);
        assert_eq!(StdoutDest::of(&as_pipeline_member, false), StdoutDest::File);

        let (captured, code) = run(StdoutDest::Capture);
        assert_eq!(
            captured.len(),
            OUTPUT_CAP,
            "the captured result stops at the cap"
        );
        assert_eq!(code, 0);
        let (written, code) = run(StdoutDest::File);
        assert_eq!(
            written.len(),
            line.len() * LINES,
            "a redirected member's file takes every match, none cut"
        );
        assert_eq!(code, 0);
    }
}

// Both platforms are driven from this host: the platform is a value. The unix rows pin the
// argv hand-off and the single-file perf gate; the Windows rows pin the scratch-file hand-off.

// One gate for the whole harness: the pin modules below share these fixtures.
#[cfg(test)]
use self::test_support::{render_only_json, serve_fixture, served, spec_cwd};

#[cfg(test)]
mod test_support {
    use super::*;

    /// Open the gate for a lane about what the engine serves, not the runtime verdict.
    pub(super) fn open_the_gate() {
        parity::publish_for_test((true, true, true));
    }

    /// The non-empty records of a search result (`--null`/`-z` terminator included),
    /// byte-sorted: the ordering-insensitive comparison the recursive rows use.
    #[cfg(target_os = "macos")]
    pub(super) fn sorted_lines(bytes: &[u8]) -> Vec<&[u8]> {
        let mut records: Vec<&[u8]> = bytes
            .split(|&b| b == b'\n' || b == b'\0')
            .filter(|record| !record.is_empty())
            .collect();
        records.sort_unstable();
        records
    }

    /// The temp tree the platform pin modules serve against: a workspace and a home beside
    /// it. Bind the TempDir for the whole test.
    pub(super) fn serve_fixture() -> (tempfile::TempDir, PathBuf, PathBuf) {
        open_the_gate();
        let tmp = tempfile::tempdir().expect("tempdir");
        let ws = tmp.path().join("ws");
        let home = tmp.path().join("home");
        fs::create_dir_all(ws.join("src")).expect("ws/src");
        fs::create_dir_all(&home).expect("home");
        fs::write(ws.join("f.txt"), "needle\n").expect("f.txt");
        fs::write(ws.join("src/main.rs"), "fn needle() {}\n").expect("main.rs");
        (tmp, ws, home)
    }

    /// Serve `command` on `platform` with the engine probe bypassed and require a rewrite with
    /// no refusal — the "this command is served" primitive both platform pin modules build on.
    /// Returns the rewrite and the scratch spec files, which the caller owns via [`SpecFiles`].
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

    /// Run the engine in-process on one spec; returns (stdout, stderr, code). `run_engine` is
    /// deliberately not used: it restores the process-wide SIGPIPE disposition (shared with
    /// tests expecting `EPIPE`) and execs the real `grep` on a cannot-serve path.
    #[cfg(unix)]
    pub(super) fn engine_run(spec: &EngineSpec) -> (Vec<u8>, Vec<u8>, i32) {
        let (out, err, code, _) = engine_run_with_stdin(spec, &[]);
        (out, err, code)
    }

    /// Run the engine in-process feeding explicit stdin bytes; returns (stdout, stderr, code,
    /// stripped stream-byte count), the last pinning the stream-size marker chain.
    #[cfg(unix)]
    pub(super) fn engine_run_with_stdin(
        spec: &EngineSpec,
        stdin: &[u8],
    ) -> (Vec<u8>, Vec<u8>, i32, Option<u64>) {
        let matcher =
            build_matcher(&spec.patterns, spec.mode, &spec.flags).expect("matcher builds");
        // The Buffer sink stands in for a pipe on fd 1, so pipeline membership decides here.
        let dest = StdoutDest::of(spec, true);
        let mut out = Output::new(OutputSink::Buffer(Vec::new()), dest);
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

    /// A spec whose rendering needs no operand on disk — for the hand-off pins that drive
    /// [`render_served`] with an executable path of their own. Returned serialized.
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

    /// The tracked cwd recorded in a Windows rewrite's spec — the cwd the engine's gate
    /// compares against. `files` is the rewrite's spec-file list.
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

    /// Serve `command` on `platform` and assert the read-only guard accepts the rewrite.
    /// Returns the rewrite and the scratch spec files the caller must drop.
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

    /// Pins the module-header invariant: a rewrite passes the guard, whichever hand-off it uses.
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

        // The operator shapes the unix word reading hands back to the shell must clear the
        // guard too: a rewrite it rejects is silently dropped.
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

        // Windows: cmd.exe cannot carry the spec, so the rewrite names a scratch file.
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

        // One scratch file per served member, all owned by the same guard.
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

    /// The serve gate follows the platform: a single-file member is served on Windows and
    /// stays on the real grep on unix.
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

    /// The shapes a read-only Windows role actually issues must clear the guard too: the
    /// engine verb is a path-qualified, quoted, extension-qualified word, and the rewrite keeps
    /// the agent's own `cd`, glob and redirect text.
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

    /// cmd.exe's cwd family is tracked the way `cd` is — `pushd <dir>` and `chdir <dir>` both
    /// change to that directory — so a search after one is served against the right cwd. The
    /// verb is read through the same folded key, so `CD` and a quoted verb are that builtin.
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

    /// A quoted `cd` target keeps its inner spaces: `cd "my dir"` tracks that directory
    /// instead of being refused, which would refuse any workspace path with a space.
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

    /// The engine receives the shell's own reading of the line, never the agent's raw text:
    /// `windows_line` has already respelled a break as `&`, so the rewrite carries no newline —
    /// the shell, not the engine, joined the members.
    #[test]
    fn the_shell_hands_over_a_line_whose_break_is_already_an_ampersand() {
        let (_tmp, ws, home) = serve_fixture();
        let (rewritten, files) = served_and_guard_accepted(
            "cd src&grep -rn needle .",
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

    /// The executable spelling a real Windows install has — a space and parentheses — must
    /// clear the guard too.
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
        .expect("the install spelling is a quotable argument")
        .text;
        let ctx = CheckContext::for_platform(&ws, ShellPlatform::Windows);
        let verdict = check_command(&fragment, &ctx);
        drop(SpecFiles(files));
        assert!(
            verdict.is_ok(),
            "read-only guard rejected the installed-executable rewrite {fragment}: {verdict:?}"
        );
    }
}

// On Windows a command carrying a grep-family invocation is either served or refused; on
// unix an unserved member keeps falling back, so nothing is refused there. Both verdicts are
// drivable from this host (the platform and probe are explicit arguments).

#[cfg(test)]
mod refusal_pins {
    use super::super::SpecFiles;
    use super::*;

    /// Serve `command` on Windows (probe bypassed) and require a refusal naming its cause.
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
        let message = unserved_failure(&cause, None);
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

    /// The probe is asked once a complete analysis has left something to serve, and a "no"
    /// demotes exactly those members: a searchless command never spawns it, and an unavailable
    /// engine leaves the analysed member unserved with that cause.
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

    /// What a Windows row's serve decision must produce.
    enum WindowsVerdict {
        /// A rewrite naming the engine, with the spec file handed over.
        Served,
        /// No rewrite and a refusal whose cause names `needle` (and `explains`, when given).
        Refused {
            needle: &'static str,
            explains: Option<&'static str>,
        },
        /// Neither served nor refused: the platform's own program runs.
        Exempt,
        /// One analysed member left unserved, with the skip's cause when named.
        Skipped { reason: Option<&'static str> },
    }

    struct WindowsCase {
        name: &'static str,
        command: &'static str,
        /// Whether the analysis is handed a home tree (a row without one pins the no-home arm).
        home: bool,
        verdict: WindowsVerdict,
        /// The trailing component the rewrite's tracked cwd must end with.
        cwd_ends: Option<&'static str>,
    }

    /// Every Windows verdict as one table: served shapes, refused shapes (with cause), and the
    /// shapes the platform leaves alone. Cosmetic variants of one rule are rows, not tests.
    #[test]
    #[expect(clippy::too_many_lines)] // one cohesive table over the Windows verdicts
    fn windows_verdicts_are_served_refused_or_exempt() {
        let (_tmp, ws, home) = serve_fixture();
        // The image and file name of this process's own path (the launcher rows compare to it).
        let exe = std::env::current_exe().expect("this process's own path");
        let image = exe.display().to_string();
        let name = exe
            .file_name()
            .expect("an executable has a file name")
            .to_string_lossy()
            .to_string();
        let cases: &[WindowsCase] = &[
            // A line that exports a new environment first is served on Windows: the engine is the
            // only search there, so the environment is not a parity question.
            WindowsCase {
                name: "a line that exports an environment first",
                command: "export LC_ALL=C && grep -rn needle .",
                home: true,
                verdict: WindowsVerdict::Served,
                cwd_ends: None,
            },
            // A lone `%` and every `!` are ordinary to `cmd /C` with `/V:OFF`.
            WindowsCase {
                name: "a lone bang in a pattern",
                command: r#"grep -rn "!=" src/main.rs"#,
                home: true,
                verdict: WindowsVerdict::Served,
                cwd_ends: None,
            },
            WindowsCase {
                name: "a lone percent in a pattern",
                command: r#"grep -n "50%" src/main.rs"#,
                home: true,
                verdict: WindowsVerdict::Served,
                cwd_ends: None,
            },
            WindowsCase {
                name: "a lone bang in a css-ish word",
                command: r#"grep -rn "!important" ."#,
                home: true,
                verdict: WindowsVerdict::Served,
                cwd_ends: None,
            },
            // The redirect shapes the runner applies by itself.
            WindowsCase {
                name: "a > with its own target word",
                command: "grep -n needle f.txt > out.txt",
                home: true,
                verdict: WindowsVerdict::Served,
                cwd_ends: None,
            },
            WindowsCase {
                name: "an append with its own target word",
                command: "grep -n needle f.txt >> out.txt",
                home: true,
                verdict: WindowsVerdict::Served,
                cwd_ends: None,
            },
            WindowsCase {
                name: "a stderr redirect with its own target word",
                command: "grep -n needle f.txt 2> err.txt",
                home: true,
                verdict: WindowsVerdict::Served,
                cwd_ends: None,
            },
            WindowsCase {
                name: "an stdin redirect with its own target word",
                command: "grep -n needle f.txt < f.txt",
                home: true,
                verdict: WindowsVerdict::Served,
                cwd_ends: None,
            },
            WindowsCase {
                name: "the stdout merge",
                command: "grep -n needle f.txt 2>&1",
                home: true,
                verdict: WindowsVerdict::Served,
                cwd_ends: None,
            },
            WindowsCase {
                name: "the stdout merge in a pipeline",
                command: "grep -n needle f.txt 2>&1 | head -2",
                home: true,
                verdict: WindowsVerdict::Served,
                cwd_ends: None,
            },
            WindowsCase {
                name: "a > inside quotes",
                command: r#"grep -n "a>b" f.txt"#,
                home: true,
                verdict: WindowsVerdict::Served,
                cwd_ends: None,
            },
            WindowsCase {
                name: "a glued > target",
                command: "grep -n needle f.txt >out.txt",
                home: true,
                verdict: WindowsVerdict::Served,
                cwd_ends: None,
            },
            WindowsCase {
                name: "a glued append target",
                command: "grep -n needle f.txt >>log.txt",
                home: true,
                verdict: WindowsVerdict::Served,
                cwd_ends: None,
            },
            WindowsCase {
                name: "a glued stderr target",
                command: "grep -n needle f.txt 2>err.txt",
                home: true,
                verdict: WindowsVerdict::Served,
                cwd_ends: None,
            },
            WindowsCase {
                name: "a glued stderr append target",
                command: "grep -n needle f.txt 2>>log.txt",
                home: true,
                verdict: WindowsVerdict::Served,
                cwd_ends: None,
            },
            WindowsCase {
                name: "a glued stdin target",
                command: "grep -n needle f.txt <in.txt",
                home: true,
                verdict: WindowsVerdict::Served,
                cwd_ends: None,
            },
            WindowsCase {
                name: "a self-contained fd redirect",
                command: "grep -n needle f.txt 1>0",
                home: true,
                verdict: WindowsVerdict::Served,
                cwd_ends: None,
            },
            // The double-quoted caret spelling is readable: the caret is inside quotes.
            WindowsCase {
                name: "a quoted caret spelling",
                command: "grep \"^fn \" src/main.rs",
                home: true,
                verdict: WindowsVerdict::Served,
                cwd_ends: None,
            },
            // The analysis needs no home on Windows: the tracked cwd resolves statically.
            WindowsCase {
                name: "a served line without a home",
                command: "cd src && grep -n needle main.rs",
                home: false,
                verdict: WindowsVerdict::Served,
                cwd_ends: Some("src"),
            },
            // A member in grep-verb position the serve decision rejected.
            WindowsCase {
                name: "an unsupported flag",
                command: "grep -P x f.txt",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "unsupported flag -P",
                    explains: None,
                },
                cwd_ends: None,
            },
            // A `;` outside quotes: cmd.exe reads it as an argument delimiter.
            WindowsCase {
                name: "an unquoted semicolon",
                command: "echo a; grep x f.txt",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "unquoted `;`",
                    explains: None,
                },
                cwd_ends: None,
            },
            // A caret outside quotes is unmodellable, and a search is present.
            WindowsCase {
                name: "a line-wide caret",
                command: "grep ^fn src/main.rs",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "the whole line",
                    explains: None,
                },
                cwd_ends: None,
            },
            // A nested member the engine never claims.
            WindowsCase {
                name: "an xargs-wrapped search",
                command: "xargs grep x f.txt",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "nested grep",
                    explains: None,
                },
                cwd_ends: None,
            },
            // A `cd` the tracked-cwd model cannot resolve.
            WindowsCase {
                name: "a switch-shaped cd target",
                command: "cd /x && grep x f.txt",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "cd untrackable",
                    explains: None,
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a popd before a search",
                command: "popd && grep -rn x .",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "cd untrackable",
                    explains: None,
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a bare pushd before a search",
                command: "pushd && grep -rn x .",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "cd untrackable",
                    explains: None,
                },
                cwd_ends: None,
            },
            // A command group cmd.exe parses itself.
            WindowsCase {
                name: "a command group",
                command: "(grep x f.txt)",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "the whole line",
                    explains: None,
                },
                cwd_ends: None,
            },
            // One servable member plus one the engine cannot serve.
            WindowsCase {
                name: "a servable member beside a nested search",
                command: "grep -rn needle . && xargs grep x f.txt",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "nested grep",
                    explains: None,
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a second grep in a pipeline",
                command: "grep -rn needle . | grep -v skip",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "nested grep",
                    explains: None,
                },
                cwd_ends: None,
            },
            // A grep inside another interpreter's command line, in every spelling of the verb.
            WindowsCase {
                name: "a cmd /c wrapped search",
                command: "cmd /c grep -rn x .",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "nested grep",
                    explains: None,
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "an upper-case cmd-wrapped search",
                command: "CMD /C GREP -rn x .",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "nested grep",
                    explains: None,
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a cmd.exe-wrapped search",
                command: "cmd.exe /c grep -rn x .",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "nested grep",
                    explains: None,
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a quoted cmd payload",
                command: r#"cmd /c "grep -rn x .""#,
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "nested grep",
                    explains: None,
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a powershell payload search",
                command: "powershell -c \"git log | grep x\"",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "nested grep",
                    explains: None,
                },
                cwd_ends: None,
            },
            // A redirect glued to a word: cmd.exe splits `x>out.txt`, while the classifier reads
            // an operator only where it OPENS the word.
            WindowsCase {
                name: "a redirect glued to the search pattern",
                command: "grep -n x>out.txt f.txt",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "redirect in `x>out.txt`",
                    explains: None,
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a redirect glued to a pattern in a walk",
                command: "grep -rn x>log .",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "redirect in `x>log`",
                    explains: None,
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a >& redirect glued to a pattern",
                command: "grep a>&echo done",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "redirect in `a>&echo`",
                    explains: None,
                },
                cwd_ends: None,
            },
            // `%…%` is expanded by cmd.exe before any program sees its argv.
            WindowsCase {
                name: "a percent pair in a pattern",
                command: "grep -n %TEMP% f.txt",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "two `%` in %TEMP%",
                    explains: None,
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a percent pair in an include filter",
                command: "grep -n x --include=%x% f.txt",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "two `%` in --include=%x%",
                    explains: None,
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a doubled percent pair",
                command: "grep -n 100%% f.txt",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "two `%` in 100%%",
                    explains: None,
                },
                cwd_ends: None,
            },
            // A redirect spelling the runner cannot apply.
            WindowsCase {
                name: "a stderr dup to the console",
                command: "grep -n needle f.txt >&2",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "`>&2`",
                    explains: Some("cannot apply the redirection"),
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "an explicit stderr dup",
                command: "grep -n needle f.txt 1>&2",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "`1>&2`",
                    explains: Some("cannot apply the redirection"),
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a third-descriptor dup",
                command: "grep -n needle f.txt 3>&1",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "`3>&1`",
                    explains: Some("cannot apply the redirection"),
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a multi-digit fd redirect with a spaced target",
                command: "grep -n needle f.txt 10> out.txt",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "`10>`",
                    explains: Some("cannot apply the redirection"),
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a multi-digit fd redirect with a glued target",
                command: "grep -n needle f.txt 10>out.txt",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "`10>out.txt`",
                    explains: Some("cannot apply the redirection"),
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a read-write redirect",
                command: "grep -n needle f.txt <>f.txt",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "`<>f.txt`",
                    explains: Some("cannot apply the redirection"),
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a bare >& dup",
                command: "grep -n needle f.txt >& out.txt",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "`>&`",
                    explains: Some("cannot apply the redirection"),
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "an append dup",
                command: "grep -n needle f.txt 2>>&1",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "`2>>&1`",
                    explains: Some("cannot apply the redirection"),
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a redirect with no target",
                command: "grep -n needle f.txt >",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "its target word is missing",
                    explains: Some("cannot apply the redirection"),
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a percent-expanded redirect target",
                command: "grep -n needle f.txt > %TMPOUT%",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "`%TMPOUT%`",
                    explains: Some("cannot apply the redirection"),
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a drive-relative redirect target",
                command: "grep -n needle f.txt > C:out.txt",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "`C:out.txt`",
                    explains: Some("cannot apply the redirection"),
                },
                cwd_ends: None,
            },
            // A directory change cmd.exe reads and the tracking cannot follow.
            WindowsCase {
                name: "a fused cd target",
                command: "cd.. && grep -rn needle .",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "`cd` member",
                    explains: None,
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a cd inside a keyword form",
                command: "if exist x cd sub && grep -rn needle .",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "keyword form",
                    explains: None,
                },
                cwd_ends: None,
            },
            WindowsCase {
                name: "a drive-relative cd target",
                command: "cd C:ws && grep -rn needle .",
                home: true,
                verdict: WindowsVerdict::Refused {
                    needle: "cd untrackable",
                    explains: None,
                },
                cwd_ends: None,
            },
            // A grep word in argument position is not the search the class refuses.
            WindowsCase {
                name: "a grep word in an argument",
                command: "echo hi & echo (grep is a tool)",
                home: true,
                verdict: WindowsVerdict::Exempt,
                cwd_ends: None,
            },
            WindowsCase {
                name: "a grep word in a parenthesised argument",
                command: "echo (grep x f.txt)",
                home: true,
                verdict: WindowsVerdict::Exempt,
                cwd_ends: None,
            },
            // The Windows exemption list: the shapes left to the platform as written.
            WindowsCase {
                name: "a line with no search",
                command: "echo hi",
                home: true,
                verdict: WindowsVerdict::Exempt,
                cwd_ends: None,
            },
            WindowsCase {
                name: "a git commit message mentioning grep",
                command: "git commit -m \"grep fix\"",
                home: true,
                verdict: WindowsVerdict::Exempt,
                cwd_ends: None,
            },
            WindowsCase {
                name: "an echoed grep word",
                command: "echo grep",
                home: true,
                verdict: WindowsVerdict::Exempt,
                cwd_ends: None,
            },
            WindowsCase {
                name: "a relative path to a grep file",
                command: r".\grep -n x f.txt",
                home: true,
                verdict: WindowsVerdict::Exempt,
                cwd_ends: None,
            },
            WindowsCase {
                name: "an absolute path to a grep file",
                command: r"C:\tools\grep.exe -rn x .",
                home: true,
                verdict: WindowsVerdict::Exempt,
                cwd_ends: None,
            },
            WindowsCase {
                name: "an upper-case git grep word",
                command: "GIT GREP x",
                home: true,
                verdict: WindowsVerdict::Exempt,
                cwd_ends: None,
            },
            WindowsCase {
                name: "a grep word glued to a leading brace",
                command: "{grep x f.txt",
                home: true,
                verdict: WindowsVerdict::Exempt,
                cwd_ends: None,
            },
            WindowsCase {
                name: "a redirect glued to the verb",
                command: "grep>x f.txt",
                home: true,
                verdict: WindowsVerdict::Exempt,
                cwd_ends: None,
            },
            WindowsCase {
                name: "an unlisted program's argument",
                command: "foo grep x",
                home: true,
                verdict: WindowsVerdict::Exempt,
                cwd_ends: None,
            },
            WindowsCase {
                name: "a find -exec grep with an escaped semicolon",
                command: r"find . -exec grep x {} \;",
                home: true,
                verdict: WindowsVerdict::Exempt,
                cwd_ends: None,
            },
            // git's own search and another machine's are recorded as skipped, never refused.
            WindowsCase {
                name: "a git grep is a skipped member",
                command: "git grep x",
                home: true,
                verdict: WindowsVerdict::Skipped { reason: None },
                cwd_ends: None,
            },
            WindowsCase {
                name: "an ssh search is a skipped member",
                command: "ssh host grep -rn x /srv",
                home: true,
                verdict: WindowsVerdict::Skipped { reason: None },
                cwd_ends: None,
            },
        ];
        let windows = ShellPlatform::Windows;
        for case in cases {
            let case_home = case.home.then_some(home.as_path());
            if let WindowsVerdict::Refused { needle, explains } = &case.verdict {
                let cause = refused(
                    case.command,
                    &ws,
                    case_home.expect("a refused row hands over a home"),
                );
                assert!(
                    cause.contains(needle),
                    "{}: reason {cause:?} lacks {needle:?}",
                    case.name
                );
                if let Some(explains) = explains {
                    assert!(
                        cause.contains(explains),
                        "{}: the cause explains the refusal: {cause:?}",
                        case.name
                    );
                }
                continue;
            }
            let mut serve = serve_command(case.command, &ws, case_home, windows, || true);
            assert!(
                serve.refusal.is_none(),
                "{}: must not be refused: {:?}",
                case.name,
                serve.refusal
            );
            match &case.verdict {
                WindowsVerdict::Served => {
                    let rewritten = serve
                        .rewritten
                        .take()
                        .unwrap_or_else(|| panic!("{}: expected a rewrite", case.name));
                    assert!(
                        rewritten.contains(ENGINE_VERB),
                        "{}: {rewritten}",
                        case.name
                    );
                    if let Some(suffix) = case.cwd_ends {
                        let cwd = spec_cwd(&serve.spec_files);
                        assert!(
                            cwd.ends_with(suffix),
                            "{}: tracked cwd {cwd:?} does not end with {suffix:?}",
                            case.name
                        );
                    }
                    drop(SpecFiles(serve.spec_files));
                }
                WindowsVerdict::Skipped { reason } => {
                    assert!(serve.rewritten.is_none(), "{}: nothing to serve", case.name);
                    assert_eq!(
                        serve.outcomes.len(),
                        1,
                        "{}: recorded as a skipped member",
                        case.name
                    );
                    assert!(!serve.outcomes[0].served, "{}: left unserved", case.name);
                    if let Some(reason) = reason {
                        assert_eq!(
                            serve.outcomes[0].reason, *reason,
                            "{}: the skip is named",
                            case.name
                        );
                    }
                }
                // Nothing to serve: Exempt, and Refused rows left the loop above.
                _ => assert!(serve.rewritten.is_none(), "{}: nothing to serve", case.name),
            }
        }

        // Two spellings of this process's own image cannot be literal rows: each names the file
        // this test binary actually has. Both are members the runner cannot run as one call.
        for (command, needle) in [
            (
                format!("start {image} -V && grep -rn needle ."),
                "runs `mahbot` itself",
            ),
            (
                format!(r".\{name} -V && grep -rn needle ."),
                "names `mahbot`'s file name",
            ),
        ] {
            let cause = refused(&command, &ws, &home);
            assert!(
                cause.contains(needle),
                "{command}: reason {cause:?} lacks {needle:?}"
            );
        }
    }

    /// A line that runs this service's own program beside a search is served like any other: a
    /// call the runner can reproduce becomes a step it spawns itself ([`plan`]), and the search
    /// beside it is served as usual.
    #[test]
    fn windows_serves_a_line_that_runs_the_own_image_beside_a_search() {
        let (_tmp, ws, home) = serve_fixture();
        let exe = std::env::current_exe().expect("this process's own path");
        let image = exe.display().to_string();
        let mut files = Vec::new();
        for command in [
            format!("{image} debug --db board && grep -rn needle ."),
            format!("grep -rn needle . && {image} --version"),
            format!("{image} -V | grep needle"),
            format!("echo hi && {image} -V && grep -rn needle ."),
        ] {
            let mut serve =
                serve_command(&command, &ws, Some(&home), ShellPlatform::Windows, || true);
            assert!(
                serve.refusal.is_none(),
                "{command}: must be served, not refused: {:?}",
                serve.refusal
            );
            assert!(serve.rewritten.is_some(), "{command}: a rewrite");
            files.append(&mut serve.spec_files);
            let plan = serve.plan.take().expect("a served rewrite has a plan");
            let own = plan
                .steps
                .iter()
                .filter(|step| matches!(step.run, plan::Run::Own { .. }))
                .count();
            assert!(own >= 1, "{command}: the image's call is an own step");
        }
        drop(SpecFiles(files));
    }

    /// What a Unix row's serve decision must produce. Unix never refuses.
    enum UnixVerdict {
        /// A rewrite naming the engine; `keeps` is a sibling it must carry verbatim.
        Served { keeps: Option<&'static str> },
        /// One analysed member left unserved with this named cause.
        Fallback(&'static str),
        /// Nothing served, with no member analysis asserted.
        Exempt,
        /// One analysed member skipped with this named cause.
        Skipped(&'static str),
    }

    struct UnixCase {
        name: &'static str,
        command: &'static str,
        home: bool,
        verdict: UnixVerdict,
    }

    /// The Unix half of the verdict table: never a refusal, fallback and served siblings pinned.
    #[test]
    #[expect(clippy::too_many_lines)] // one cohesive table over the Unix verdicts
    fn unix_never_refuses_and_keeps_the_fallback() {
        let (_tmp, ws, home) = serve_fixture();
        let cases: &[UnixCase] = &[
            // A line that exports a new environment before its search is left to the real `grep`.
            UnixCase {
                name: "an export before a search",
                command: "export LC_ALL=C; grep -rn needle .",
                home: true,
                verdict: UnixVerdict::Fallback("the line changes the environment first"),
            },
            UnixCase {
                name: "an export joined by &&",
                command: "export LC_ALL=C && grep -rn needle .",
                home: true,
                verdict: UnixVerdict::Fallback("the line changes the environment first"),
            },
            UnixCase {
                name: "an unset before a search",
                command: "unset LANG; grep -rn needle .",
                home: true,
                verdict: UnixVerdict::Fallback("the line changes the environment first"),
            },
            UnixCase {
                name: "an eval export before a search",
                command: "eval \"export LC_ALL=C\"; grep -rn needle .",
                home: true,
                verdict: UnixVerdict::Fallback("the line changes the environment first"),
            },
            UnixCase {
                name: "a source before a search",
                command: "source /dev/null; grep -rn needle .",
                home: true,
                verdict: UnixVerdict::Fallback("the line changes the environment first"),
            },
            UnixCase {
                name: "a quoted builtin before a search",
                command: "\"export\" LC_ALL=C; grep -rn needle .",
                home: true,
                verdict: UnixVerdict::Fallback("the line changes the environment first"),
            },
            UnixCase {
                name: "an escaped builtin before a search",
                command: "\\export LC_ALL=C; grep -rn needle .",
                home: true,
                verdict: UnixVerdict::Fallback("the line changes the environment first"),
            },
            UnixCase {
                name: "a subshell export before a search",
                command: "(export LC_ALL=C); grep -rn needle .",
                home: true,
                verdict: UnixVerdict::Fallback("the line changes the environment first"),
            },
            UnixCase {
                name: "a lone assignment before a search",
                command: "LC_ALL=C; grep -rn needle .",
                home: true,
                verdict: UnixVerdict::Fallback("the line changes the environment first"),
            },
            UnixCase {
                name: "a path assignment before a search",
                command: "PATH=/opt/bin:$PATH; grep -rn needle .",
                home: true,
                verdict: UnixVerdict::Fallback("the line changes the environment first"),
            },
            UnixCase {
                name: "an assignment prefixed to a builtin",
                command: "LC_ALL=C export FOO=1; grep -rn needle .",
                home: true,
                verdict: UnixVerdict::Fallback("the line changes the environment first"),
            },
            UnixCase {
                name: "an export inside a compound body",
                command: "if true; then export LC_ALL=C; fi; grep -rn needle .",
                home: true,
                verdict: UnixVerdict::Fallback("the line changes the environment first"),
            },
            // A search before the export runs in the product's environment and is served.
            UnixCase {
                name: "a search before an export",
                command: "grep -rn needle . ; export LC_ALL=C",
                home: true,
                verdict: UnixVerdict::Served { keeps: None },
            },
            UnixCase {
                name: "a search after a cd",
                command: "cd . && grep -rn needle .",
                home: true,
                verdict: UnixVerdict::Served { keeps: None },
            },
            UnixCase {
                name: "a search after an unrelated command",
                command: "echo hi; grep -rn needle .",
                home: true,
                verdict: UnixVerdict::Served { keeps: None },
            },
            UnixCase {
                name: "an assignment-looking argument before a search",
                command: "echo KEY=value; grep -rn needle .",
                home: true,
                verdict: UnixVerdict::Served { keeps: None },
            },
            UnixCase {
                name: "a KEY=value pattern",
                command: "grep -rn KEY=value .",
                home: true,
                verdict: UnixVerdict::Served { keeps: None },
            },
            UnixCase {
                name: "an export in a pipeline head",
                command: "export LC_ALL=C | grep needle",
                home: true,
                verdict: UnixVerdict::Served { keeps: None },
            },
            UnixCase {
                name: "an assignment in a pipeline head",
                command: "LC_ALL=C echo hi | grep needle",
                home: true,
                verdict: UnixVerdict::Served { keeps: None },
            },
            // The servable member is still served while the rejected sibling runs the real `grep`.
            UnixCase {
                name: "a served sibling of a rejected member",
                command: "grep -rn needle . && xargs grep x f.txt",
                home: true,
                verdict: UnixVerdict::Served {
                    keeps: Some("xargs grep x f.txt"),
                },
            },
            // The `%…%` reading is the cmd.exe one alone.
            UnixCase {
                name: "a percent pair on unix",
                command: "grep -rn %TEMP% .",
                home: true,
                verdict: UnixVerdict::Served { keeps: None },
            },
            UnixCase {
                name: "a percent pair in an include filter on unix",
                command: "grep -rn x --include=%x% .",
                home: true,
                verdict: UnixVerdict::Served { keeps: None },
            },
            UnixCase {
                name: "a bang in a quoted pattern on unix",
                command: "grep -rn \"a!b\" .",
                home: true,
                verdict: UnixVerdict::Served { keeps: None },
            },
            // The rows Windows refuses are the unix fallback: the real `grep` keeps running.
            UnixCase {
                name: "an unsupported flag",
                command: "grep -P x f.txt",
                home: true,
                verdict: UnixVerdict::Exempt,
            },
            UnixCase {
                name: "an unquoted semicolon",
                command: "echo a; grep x f.txt",
                home: true,
                verdict: UnixVerdict::Exempt,
            },
            UnixCase {
                name: "a line-wide caret",
                command: "grep ^fn src/main.rs",
                home: true,
                verdict: UnixVerdict::Exempt,
            },
            UnixCase {
                name: "an xargs-wrapped search",
                command: "xargs grep x f.txt",
                home: true,
                verdict: UnixVerdict::Exempt,
            },
            UnixCase {
                name: "a cd the tracking cannot follow",
                command: "cd /x && grep x f.txt",
                home: true,
                verdict: UnixVerdict::Exempt,
            },
            UnixCase {
                name: "a command group",
                command: "(grep x f.txt)",
                home: true,
                verdict: UnixVerdict::Exempt,
            },
            UnixCase {
                name: "a git grep",
                command: "git grep x",
                home: true,
                verdict: UnixVerdict::Exempt,
            },
            // Without a home nothing that could expand `~` is served, and nothing is refused.
            UnixCase {
                name: "a search without a home",
                command: "grep -rn needle .",
                home: false,
                verdict: UnixVerdict::Exempt,
            },
            // A wrapped interpreter is a skipped member on unix too.
            UnixCase {
                name: "a cmd-wrapped search",
                command: "cmd /c grep x f.txt",
                home: true,
                verdict: UnixVerdict::Skipped("nested grep"),
            },
        ];
        for case in cases {
            let case_home = case.home.then_some(home.as_path());
            let mut serve =
                serve_command(case.command, &ws, case_home, ShellPlatform::Unix, || true);
            assert!(
                serve.refusal.is_none(),
                "{}: unix never refuses: {:?}",
                case.name,
                serve.refusal
            );
            match &case.verdict {
                UnixVerdict::Served { keeps } => {
                    let rewritten = serve
                        .rewritten
                        .take()
                        .unwrap_or_else(|| panic!("{}: expected a rewrite", case.name));
                    assert!(
                        rewritten.contains(ENGINE_VERB),
                        "{}: {rewritten}",
                        case.name
                    );
                    if let Some(keeps) = keeps {
                        assert!(
                            rewritten.contains(keeps),
                            "{}: the sibling is kept verbatim: {rewritten}",
                            case.name
                        );
                    }
                    drop(SpecFiles(serve.spec_files));
                }
                UnixVerdict::Fallback(reason) => {
                    assert!(
                        serve.rewritten.is_none(),
                        "{}: nothing to serve (got {:?})",
                        case.name,
                        serve.rewritten
                    );
                    assert_eq!(
                        serve.outcomes.len(),
                        1,
                        "{}: the member is analysed",
                        case.name
                    );
                    assert!(
                        !serve.outcomes[0].served,
                        "{}: the member falls back",
                        case.name
                    );
                    assert_eq!(
                        serve.outcomes[0].reason, *reason,
                        "{}: the cause is named",
                        case.name
                    );
                }
                UnixVerdict::Exempt => {
                    assert!(
                        serve.rewritten.is_none(),
                        "{}: nothing to serve (got {:?})",
                        case.name,
                        serve.rewritten
                    );
                }
                UnixVerdict::Skipped(reason) => {
                    assert!(
                        serve.rewritten.is_none(),
                        "{}: nothing to serve (got {:?})",
                        case.name,
                        serve.rewritten
                    );
                    assert_eq!(
                        serve.outcomes.len(),
                        1,
                        "{}: recorded as a skipped member",
                        case.name
                    );
                    assert!(!serve.outcomes[0].served, "{}: left unserved", case.name);
                    assert_eq!(
                        serve.outcomes[0].reason, *reason,
                        "{}: the skip is named",
                        case.name
                    );
                }
            }
        }
    }
}

// The rewrite and the plan are two renderings of one per-member decision. The guard is that
// they agree: a plan whose members were not the line's would report output from somewhere the
// agent did not ask about.

#[cfg(test)]
mod plan_pins {
    use super::super::SpecFiles;
    use super::super::plan::{Plan, Run};
    use super::*;

    /// The rewrite, the plan and the scratch spec files for `command` on Windows, probe bypassed.
    /// The platform and probe are arguments, so the decomposition is drivable from this host.
    fn served_with_plan(command: &str, ws: &Path, home: &Path) -> (String, Plan, Vec<PathBuf>) {
        let mut serve = serve_command(command, ws, Some(home), ShellPlatform::Windows, || true);
        assert!(
            serve.refusal.is_none(),
            "{command}: served, not refused: {:?}",
            serve.refusal
        );
        let rewritten = serve
            .rewritten
            .take()
            .unwrap_or_else(|| panic!("{command}: expected a rewrite"));
        let plan = serve
            .plan
            .take()
            .unwrap_or_else(|| panic!("{command}: a produced Windows rewrite carries a plan"));
        (rewritten, plan, serve.spec_files)
    }

    /// One plan member, flattened out of its step: a fold's text splits back into the members
    /// cmd.exe reads, an own-image step is compared by the argv it is spawned with.
    enum Planned {
        Shell(String),
        Own(Vec<String>, Vec<String>),
    }

    /// The plan's members with the connector that follows each — what the rewrite must spell.
    fn planned_members(plan: &Plan) -> Vec<(Planned, String)> {
        let mut members = Vec::new();
        for (index, step) in plan.steps.iter().enumerate() {
            // The connector after a step is the next step's own join.
            let after = plan
                .steps
                .get(index + 1)
                .map_or("", |next| next.join.spelling());
            match &step.run {
                Run::Shell { text } => {
                    let fold = windows::segment_command(text).expect("cmd.exe reads a fold");
                    let last = fold.len() - 1;
                    for (at, (member, connector)) in fold.into_iter().enumerate() {
                        members.push((
                            Planned::Shell(member),
                            if at == last {
                                after.to_string()
                            } else {
                                connector
                            },
                        ));
                    }
                }
                Run::Own { args, redirects } => members.push((
                    Planned::Own(args.clone(), redirects.clone()),
                    after.to_string(),
                )),
            }
        }
        members
    }

    /// A cwd a plan case expects, resolved against the fixture's workspace.
    #[derive(Clone, Copy)]
    enum ExpectedCwd {
        Root,
        Src,
    }

    struct PlanCase {
        name: &'static str,
        command: String,
        /// The cwd every step must carry, when the row pins them.
        cwds: Option<&'static [ExpectedCwd]>,
    }

    /// The drift guard: the rendered rewrite IS the plan, read back with cmd.exe's own model.
    /// Every fold's members are byte-identical to the rewrite's, every own-image step delivers
    /// exactly the argv the runner spawns (`[exe] ++ args ++ redirects`), the connectors are the
    /// plan's, and every step's cwd is the one the analyzer tracked.
    ///
    /// One exception, the divergence [`plan`] names: a member carried verbatim keeps the
    /// agent's spelling of the image (`mahbot -V`) while the plan spawns the absolute path, so
    /// that word is compared by the file it names and everything after it byte for byte.
    #[test]
    #[expect(clippy::too_many_lines)] // one cohesive table over the rewrite/plan pairs
    fn the_plan_matches_the_rewritten_line() {
        let (_tmp, ws, home) = serve_fixture();
        let root = canonical_or_lexical(&ws, ShellPlatform::Windows);
        let src = root.join("src");
        // The image as this process spells it: the test binary's own file name.
        let exe = std::env::current_exe().expect("this process's own path");
        let name = exe
            .file_name()
            .expect("an executable has a file name")
            .to_string_lossy()
            .to_string();
        let bare_call = format!("{name} -V && grep -rn needle .");
        let prefixed_call = format!("@{name} -V && grep -rn needle .");
        let cases: Vec<PlanCase> = vec![
            PlanCase {
                name: "a lone search",
                command: "grep -rn needle .".to_string(),
                cwds: Some(&[ExpectedCwd::Root]),
            },
            PlanCase {
                name: "a pipelined search",
                command: "grep -rn needle . | head -3".to_string(),
                cwds: None,
            },
            PlanCase {
                name: "a cd before a search",
                command: "cd src && grep -rn needle .".to_string(),
                cwds: Some(&[ExpectedCwd::Root, ExpectedCwd::Src]),
            },
            PlanCase {
                name: "a cd search and tail",
                command: "cd src && grep -rn needle . | head -3".to_string(),
                cwds: Some(&[ExpectedCwd::Root, ExpectedCwd::Src, ExpectedCwd::Src]),
            },
            // The `cd` is inside the fold that follows the search, so it starts in the line's cwd.
            PlanCase {
                name: "a cd in the fold after a search",
                command: "grep -rn needle . && cd src && echo after".to_string(),
                cwds: Some(&[ExpectedCwd::Root, ExpectedCwd::Root]),
            },
            PlanCase {
                name: "two searches",
                command: "grep needle f.txt && grep needle src/main.rs".to_string(),
                cwds: Some(&[ExpectedCwd::Root, ExpectedCwd::Root]),
            },
            PlanCase {
                name: "a search with a redirect",
                command: "grep -n needle f.txt > out.txt".to_string(),
                cwds: None,
            },
            PlanCase {
                name: "a search with an append redirect",
                command: "grep -n needle f.txt >> out.txt".to_string(),
                cwds: None,
            },
            PlanCase {
                name: "a search with the stdout merge",
                command: "grep -n needle f.txt 2>&1".to_string(),
                cwds: None,
            },
            PlanCase {
                name: "a search with the stdout merge in a pipeline",
                command: "grep -n needle f.txt 2>&1 | head -2".to_string(),
                cwds: None,
            },
            PlanCase {
                name: "a shell member before a search",
                command: "type note.txt && grep -n needle f.txt".to_string(),
                cwds: Some(&[ExpectedCwd::Root, ExpectedCwd::Root]),
            },
            PlanCase {
                name: "a bare-name call of the image",
                command: bare_call,
                cwds: None,
            },
            PlanCase {
                name: "an @-prefixed call of the image",
                command: prefixed_call,
                cwds: None,
            },
        ];
        for case in cases {
            let command = case.command.as_str();
            let (rewritten, plan, files) = served_with_plan(command, &ws, &home);
            let written = windows::segment_command(&rewritten).expect("cmd.exe reads the rewrite");
            let planned = planned_members(&plan);
            assert_eq!(
                written.len(),
                planned.len(),
                "{}: the plan and the rewrite must have the same members \
                 (rewrite: {rewritten})",
                case.name
            );
            let plan_exe = plan.exe.to_string_lossy();
            for (at, ((written_text, written_conn), (member, connector))) in
                written.iter().zip(&planned).enumerate()
            {
                assert_eq!(
                    written_conn, connector,
                    "{}: connector before {at}",
                    case.name
                );
                match member {
                    Planned::Shell(text) => assert_eq!(
                        written_text, text,
                        "{}: the fold's member {at} must be verbatim",
                        case.name
                    ),
                    Planned::Own(args, redirects) => {
                        let delivered: Vec<String> = windows::tokenize(written_text)
                            .expect("cmd.exe reads the member")
                            .iter()
                            .map(|word| windows::unquote_word(&word.raw))
                            .collect();
                        let expected: Vec<String> = std::iter::once(plan_exe.to_string())
                            .chain(args.iter().cloned())
                            .chain(redirects.iter().map(|token| windows::unquote_word(token)))
                            .collect();
                        if delivered != expected {
                            // The named divergence: a verbatim member keeps the agent's spelling of
                            // the image while the plan spawns the absolute path. The word must name
                            // the same file and every word after it must be identical.
                            assert_eq!(
                                delivered.len(),
                                expected.len(),
                                "{}: member {at}",
                                case.name
                            );
                            assert_eq!(delivered[1..], expected[1..], "{}: member {at}", case.name);
                            let named = delivered[0]
                                .trim_start_matches(['@', '('])
                                .trim_end_matches(')');
                            assert_eq!(
                                std::path::Path::new(named).file_name(),
                                std::path::Path::new(&expected[0]).file_name(),
                                "{}: member {at} names another file: {written_text}",
                                case.name
                            );
                        }
                    }
                }
            }
            if let Some(cwds) = case.cwds {
                let expected: Vec<PathBuf> = cwds
                    .iter()
                    .map(|cwd| match cwd {
                        ExpectedCwd::Root => root.clone(),
                        ExpectedCwd::Src => src.clone(),
                    })
                    .collect();
                let steps: Vec<&Path> = plan.steps.iter().map(|step| step.cwd.as_path()).collect();
                assert_eq!(
                    steps,
                    expected.iter().map(PathBuf::as_path).collect::<Vec<_>>(),
                    "{}: step cwds",
                    case.name
                );
            }
            drop(SpecFiles(files));
        }
        // A plan exists only where a rewrite does: a demoted command carries none.
        for (name, command) in [
            ("a servable search", "grep -rn needle ."),
            ("an unservable search", "grep -P needle f.txt"),
        ] {
            let serve = serve_command(command, &ws, Some(&home), ShellPlatform::Windows, || false);
            assert!(serve.plan.is_none(), "{name}: a demotion carries no plan");
            assert!(serve.rewritten.is_none(), "{name}: no rewrite");
        }
        // Nothing is planned on unix: the shell waits for its children there.
        let unix = serve_command(
            "grep -rn needle .",
            &ws,
            Some(&home),
            ShellPlatform::Unix,
            || true,
        );
        assert!(unix.plan.is_none(), "unix carries no plan");
        assert!(unix.rewritten.is_some(), "unix still serves");
    }
}

// The reading the module header states, observed from both sides: which text the member is
// served on, and that the operator is handed back to the shell to perform.

#[cfg(all(test, unix))]
mod unix_operator_pins {
    use super::test_support::engine_run;
    use super::*;
    use std::process::Command;

    /// What the served member was read as, in one labelled line: the text it searches, the raw
    /// redirect spellings returned to the shell, and its operands. A mismatch names its field.
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

    const ENGINE_OUT: &str = "ENGINE-OUT";
    const ENGINE_ERR: &str = "ENGINE-ERR";
    /// The stand-in's own exit status: a row can tell whether the shell waited for the member.
    const ENGINE_EXIT: i32 = 7;

    /// A stand-in for the engine binary the rewrite names, answering on both streams so a row
    /// can see which stream the shell routed where.
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

    /// One word-reading row: the served member's search, redirects and operands.
    struct ReadingCase {
        name: &'static str,
        command: &'static str,
        expected: &'static str,
    }

    /// The unix reading's cases as one table: `sh`'s comment, the glued-operator family, the
    /// text actually searched, and process substitution.
    #[test]
    #[expect(clippy::too_many_lines)] // one cohesive table over the reading's cases
    fn the_unix_word_reading_cases() {
        let (tmp, ws, home) = serve_fixture();
        let cases: &[ReadingCase] = &[
            // An unquoted `#` opening a word comments out the rest of its line.
            ReadingCase {
                name: "a comment takes an ampersand and an echo with it",
                command: "grep -rn needle . # note & echo TAIL-RAN",
                expected: "search=needle redirect= operands=.",
            },
            ReadingCase {
                name: "an unbalanced quote inside a comment is inert",
                command: "grep -rn needle . # don't",
                expected: "search=needle redirect= operands=.",
            },
            // A `#` that opens no word stays ordinary text (mid-word, quoted, escaped).
            ReadingCase {
                name: "a mid-word hash is ordinary text",
                command: "grep -rn a#b .",
                expected: "search=a#b redirect= operands=.",
            },
            ReadingCase {
                name: "a quoted hash is ordinary text",
                command: "grep -rn 'a#b' .",
                expected: "search=a#b redirect= operands=.",
            },
            ReadingCase {
                name: "an escaped hash is ordinary text",
                command: r"grep -rn a\#b .",
                expected: "search=a#b redirect= operands=.",
            },
            // The glued-operator family, read the way `sh` reads it.
            ReadingCase {
                name: "a separated > target",
                command: "grep -rn needle > out.txt .",
                expected: "search=needle redirect=> out.txt operands=.",
            },
            ReadingCase {
                name: "a separated stderr target",
                command: "grep -rn needle 2> out.txt .",
                expected: "search=needle redirect=2> out.txt operands=.",
            },
            ReadingCase {
                name: "a word glued to a > operator",
                command: "grep -rn needle>out.txt .",
                expected: "search=needle redirect=>out.txt operands=.",
            },
            ReadingCase {
                name: "a word glued to an append operator",
                command: "grep -rn needle>>out.txt .",
                expected: "search=needle redirect=>>out.txt operands=.",
            },
            ReadingCase {
                name: "a word glued to an stdin operator",
                command: "grep -rn needle<f.txt .",
                expected: "search=needle redirect=<f.txt operands=.",
            },
            ReadingCase {
                name: "a word glued to a stdout dup",
                command: "grep -rn needle>&1 .",
                expected: "search=needle redirect=>&1 operands=.",
            },
            // `<&` is the input-side fd dup: one operator too.
            ReadingCase {
                name: "a separated input-side dup",
                command: "grep -rn needle . <&2",
                expected: "search=needle redirect=<&2 operands=.",
            },
            ReadingCase {
                name: "an fd-prefixed input-side dup",
                command: "grep -rn 2<&1 needle .",
                expected: "search=needle redirect=2<&1 operands=.",
            },
            // `&>` is one operator, and the word before it stays part of the search.
            ReadingCase {
                name: "a word glued to a &> operator",
                command: "grep -rn needle&>out.txt .",
                expected: "search=needle redirect=&>out.txt operands=.",
            },
            ReadingCase {
                name: "a word glued to an append &> operator",
                command: "grep -rn needle&>>out.txt .",
                expected: "search=needle redirect=&>>out.txt operands=.",
            },
            // The digits belong to the operator only when everything before it
            // is digits.
            ReadingCase {
                name: "a digit-prefixed redirect",
                command: "grep -rn 2>out.txt needle .",
                expected: "search=needle redirect=2>out.txt operands=.",
            },
            ReadingCase {
                name: "a multi-digit-prefixed redirect",
                command: "grep -rn 10>out.txt needle .",
                expected: "search=needle redirect=10>out.txt operands=.",
            },
            ReadingCase {
                name: "a digit ending a word before a redirect",
                command: "grep -rn needle2>out.txt .",
                expected: "search=needle2 redirect=>out.txt operands=.",
            },
            // Quoted and escaped operators are ordinary text.
            ReadingCase {
                name: "a single-quoted operator is text",
                command: "grep -rn 'a>b' .",
                expected: "search=a>b redirect= operands=.",
            },
            ReadingCase {
                name: "a double-quoted operator is text",
                command: "grep -rn \"x>log\" .",
                expected: "search=x>log redirect= operands=.",
            },
            ReadingCase {
                name: "an escaped operator is text",
                command: r"grep -rn a\>b .",
                expected: "search=a>b redirect= operands=.",
            },
        ];
        for case in cases {
            assert_eq!(
                reading(case.command, &ws, &home),
                case.expected,
                "{}: {}",
                case.name,
                case.command
            );
        }

        // The same search spelled with the operator separate: the words `sh` would have passed,
        // with the operator gone from the argv.
        let expected = "f.txt:needle\nsrc/main.rs:fn needle() {}\n";
        for (name, command) in [
            ("the separated spelling", "grep needle f.txt src/main.rs"),
            (
                "a glued > spelling",
                "grep needle>out.txt f.txt src/main.rs",
            ),
            (
                "a glued append spelling",
                "grep needle>>out.txt f.txt src/main.rs",
            ),
            (
                "a glued stdin spelling",
                "grep needle<f.txt f.txt src/main.rs",
            ),
            (
                "a glued &> spelling",
                "grep needle&>out.txt f.txt src/main.rs",
            ),
            (
                "an fd-prefixed spelling",
                "grep 2>out.txt needle f.txt src/main.rs",
            ),
        ] {
            let analyzed = analyze_command(command, &ws, &home, ShellPlatform::Unix, false)
                .unwrap_or_else(|e| panic!("{name}: expected servable, got {}", e.reason));
            let (out, err, code) = engine_run(&analyzed.specs[0]);
            assert_eq!(code, 0, "{name}: exit");
            assert_eq!(
                String::from_utf8_lossy(&out),
                expected,
                "{name}: searched text"
            );
            assert!(err.is_empty(), "{name}: stderr: {err:?}");
        }

        // `>(…)`/`<(…)` is process substitution, never a redirection: the member stays real.
        for (name, command) in [
            (
                "a process-substitution input",
                "grep -rn needle <(cat f.txt) .",
            ),
            (
                "a process-substitution output",
                "grep -rn needle . >(cat f.txt)",
            ),
        ] {
            let serve = serve_command(command, &ws, Some(&home), ShellPlatform::Unix, || true);
            assert!(serve.refusal.is_none(), "{name}: unix never refuses");
            assert!(
                serve.rewritten.is_none(),
                "{name}: nothing to serve (got {:?})",
                serve.rewritten
            );
            assert_eq!(serve.outcomes.len(), 1, "{name}: one analyzed member");
            assert_eq!(serve.outcomes[0].reason, "process substitution", "{name}");
        }
        drop(tmp);
    }

    /// The shell's side of the reading: the operators are performed by a real `sh`.
    #[test]
    #[expect(clippy::too_many_lines)] // one cohesive table over the shell's cases
    fn the_shell_performs_the_operators() {
        let (tmp, ws, home) = serve_fixture();
        let engine = engine_stub(tmp.path());

        // The comment's side: nothing it swallowed runs, its `&` never backgrounds the member.
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
            let rewritten = join_rewritten(
                &analyzed.segments,
                &analyzed.specs,
                &jsons,
                ShellPlatform::Unix,
                &engine,
                &analyzed.root_cwd,
            )
            .expect("the unix join never refuses")
            .text;
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

        // The redirect lands in the file it named, where `sh` would have written it.
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
            let rewritten = join_rewritten(
                &analyzed.segments,
                &analyzed.specs,
                &jsons,
                ShellPlatform::Unix,
                &engine,
                &analyzed.root_cwd,
            )
            .expect("the unix join never refuses")
            .text;
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

        // A member backgrounded last in its line: the call returns on the shell's own status.
        let command = "grep -rn needle . &";
        {
            let analyzed = analyze_command(command, &ws, &home, ShellPlatform::Unix, false)
                .unwrap_or_else(|e| panic!("{command}: expected servable, got {}", e.reason));
            let jsons: Vec<String> = analyzed.specs.iter().map(spec_json).collect();
            let rewritten = join_rewritten(
                &analyzed.segments,
                &analyzed.specs,
                &jsons,
                ShellPlatform::Unix,
                &engine,
                &analyzed.root_cwd,
            )
            .expect("the unix join never refuses")
            .text;
            assert!(
                rewritten.ends_with(" &"),
                "{command}: the `&` rides the rewrite: {rewritten}"
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
                "{command}: the shell's status, not the backgrounded member's: {rewritten}"
            );
        }
    }
}
