//! The runtime locale-parity gate for the grep engine's fast served path.
//!
//! The engine's matching model is its own — Rust regex over UTF-8, with the C.UTF-8
//! deltas the module header lists — while the search it stands in for is a locale-aware
//! `grep` running in the environment the owner's terminal gives the agent's command
//! ([`super::super::agent_env_pairs`]). That locale is unknown to the product, so the
//! engine may no longer assume the `LC_ALL=C.UTF-8` it was written against and a serve
//! decision cannot rest on the engine's own say-so.
//!
//! This module publishes a coarse, runtime-measured verdict ([`Established`], read through
//! [`serve_allowed`]) that the parent-side serve path consults on every member: the fast
//! path may stand only where the product established, under the locale actually in effect,
//! that the engine agrees with the real search for that member's shape ([`Need`]).
//! Otherwise the real search runs; where there is no real search at all — Windows, or a
//! unix environment whose search path resolves none — the fast path is the only search.
//!
//! The verdict comes from a bounded differential battery ([`measure`]) over a byte-exact
//! fixture: the same rows through the production parent path in-process and through the
//! real `grep`/`sh` under the agent's environment, a capability being established only
//! when every compared row agreed (stdout bytes and exit code; sorted record sets for
//! recursive walks). Only booleans, counts and the probe's finding are logged — never a
//! locale name, never anything else read from the environment.
//!
//! The rows deliberately include the engine's approved deltas (Unicode simple case
//! folding that makes `-i k` match KELVIN, ASCII-only POSIX classes, the C.UTF-8 reading
//! of invalid UTF-8), so [`Need::General`] is *expected* to stay unestablished on a host
//! whose `grep` folds case by lowercase comparison, and a `C` locale is expected to
//! unestablish [`Need::Plain`] too. That is the point: the fast path kept for a shape is
//! the shape this host's own search was measured to agree with.
//!
//! A verdict belongs to the environment it was measured under: the refresher
//! ([`refresh_if_environment_changed`]) drops the published one before measuring a changed
//! environment — normally in the reader's own step, before the new environment is in place
//! — so the window a battery takes is served fail-closed by the real search.

use std::ffi::OsString;
use std::io::{self, Read, Seek};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use arc_swap::ArcSwapOption;

use crate::tools::path::shell_quote;
use crate::util::UnwrapPoison;

use super::super::ShellPlatform;
use super::{
    EngineSpec, MatchMode, Output, OutputSink, ParsedGrep, PipelineCtx, StdoutDest, build_matcher,
    expand_glob, has_unquoted_glob, serve_into, serve_one_grep,
};

/// Deadline on one battery child: generous against the milliseconds a real `grep` or `sh`
/// takes, and small enough that an owner-`PATH` program which never finishes cannot wedge
/// the measurement, its blocking thread or the environment reader that waits for it.
const BATTERY_SPAWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Cap on one battery child's stdout, both read back and writable before it is killed: far
/// above anything a comparison needs, bounding what a runaway owner-`PATH` program can cost
/// the reader's private temp root and memory.
const BATTERY_STDOUT_CAP: u64 = 1024 * 1024;

/// The one thing a member's shape is judged by, deliberately coarse. A served search agrees
/// with the real one either because its matching cannot involve a locale at all
/// ([`Need::Plain`]) or because it can and the model was proven under the locale in effect
/// ([`Need::General`]); glob operands and filters add a dimension of their own
/// ([`Need::Glob`]) because the engine globs with libc `fnmatch` from a process that never
/// calls `setlocale`, while the shell and grep glob under the locale they were handed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Need {
    /// A literal, case-sensitive byte search: no `-i`, no `-w`, and no regex metacharacter
    /// or backslash in any raw pattern. The one shape whose *engine-side* matching has no
    /// locale input, so it is worth measuring rather than assuming — the real search does
    /// not agree byte for byte under every locale (a `C` locale reads an invalid-UTF-8 line
    /// the engine treats as a non-match).
    Plain,
    /// Everything else — case-insensitive matching, POSIX classes, `.`, brackets, `-w`: the
    /// engine's model differs from a locale-aware grep's, so it needs a proof.
    General,
    /// The operands carry an unquoted glob, or the member carries
    /// `--include`/`--exclude`/`--exclude-dir` filters: `fnmatch` is locale-sensitive.
    Glob,
}

impl Need {
    /// The capability's own word, used as the demotion's fallback reason.
    const fn capability(self) -> &'static str {
        match self {
            Need::Plain => "literal matching",
            Need::General => "pattern matching",
            Need::Glob => "glob expansion",
        }
    }
}

/// The capabilities one member's shape needs: exactly one of [`Need::Plain`]/
/// [`Need::General`], plus [`Need::Glob`] when its operands or filters carry a glob surface.
/// Fail-closed by construction: anything the reading does not positively recognise as a
/// byte-literal shape is [`Need::General`].
#[must_use]
pub(super) fn needs(parsed: &ParsedGrep, platform: ShellPlatform) -> Vec<Need> {
    let mut needs = Vec::with_capacity(2);
    needs.push(if is_byte_literal(parsed) {
        Need::Plain
    } else {
        Need::General
    });
    if has_glob_surface(parsed, platform) {
        needs.push(Need::Glob);
    }
    needs
}

/// Whether the member's matching cannot depend on either search's locale: every raw pattern
/// is a literal in every dialect (`-F` by definition; otherwise a metacharacter or backslash
/// disqualifies it) and no flag puts case or word boundaries between pattern and bytes.
/// What the *engine* does with such a pattern has no locale input; what the real search does
/// with it is what the battery measures before the fast path may stand.
fn is_byte_literal(parsed: &ParsedGrep) -> bool {
    if parsed.flags.i || parsed.flags.w {
        return false;
    }
    // `-F` patterns are fixed strings whatever they contain; the engine escapes them into a
    // literal matcher, and the real search compares bytes.
    parsed.mode == MatchMode::Fixed
        || parsed.patterns.iter().all(|pattern| {
            !pattern.contains([
                '.', '^', '$', '*', '+', '?', '(', ')', '[', ']', '{', '}', '|', '\\',
            ])
        })
}

/// Whether the member carries a glob surface the shell would expand under its own locale:
/// an unquoted glob in an operand token, or an include/exclude filter.
fn has_glob_surface(parsed: &ParsedGrep, platform: ShellPlatform) -> bool {
    !parsed.filters.is_empty()
        || !parsed.exclude_dir.is_empty()
        || parsed
            .operand_tokens
            .iter()
            .any(|tok| has_unquoted_glob(tok, platform))
}

/// What the battery established for the locale it was measured under, one flag per [`Need`].
/// `Default` is all false: fail-closed until measured.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Established {
    plain: bool,
    general: bool,
    glob: bool,
}

impl Established {
    const fn all() -> Self {
        Established {
            plain: true,
            general: true,
            glob: true,
        }
    }

    const fn covers(self, need: Need) -> bool {
        match need {
            Need::Plain => self.plain,
            Need::General => self.general,
            Need::Glob => self.glob,
        }
    }
}

/// The verdict the serve path reads, lock-free, on every member.
static ESTABLISHED: ArcSwapOption<Established> = ArcSwapOption::const_empty();

// A per-thread verdict override, installed by test lanes: a lane publishing a verdict
// describes its own runs only, and a process-global publish would race sibling lanes.
#[cfg(test)]
thread_local! {
    static TEST_VERDICT: std::cell::Cell<Option<Established>> =
        const { std::cell::Cell::new(None) };
}

/// The verdict in effect: a test thread's override when installed, else the measured global.
fn established() -> Established {
    #[cfg(test)]
    if let Some(verdict) = TEST_VERDICT.with(std::cell::Cell::get) {
        return verdict;
    }
    ESTABLISHED.load().as_deref().copied().unwrap_or_default()
}

/// Publish a measured verdict; the next [`established`] read sees it.
fn publish(verdict: Established) {
    ESTABLISHED.store(Some(Arc::new(verdict)));
}

// Whether the calling thread is inside the battery's own row building. The battery builds
// each row's spec through the production parent path, which consults this same gate: without
// the flag, a fail-closed verdict would demote the very rows the verdict is measured from.
thread_local! {
    static MEASURING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Suspends the gate for its lifetime; the drop covers a panic, so a thread is never left open.
struct Measuring;

impl Measuring {
    fn enter() -> Self {
        MEASURING.with(|flag| flag.set(true));
        Measuring
    }
}

impl Drop for Measuring {
    fn drop(&mut self) {
        MEASURING.with(|flag| flag.set(false));
    }
}

/// The environment the verdict in effect was measured for: set by [`record`] with the
/// verdict, cleared by [`invalidate_verdict_for`] before a command can see the new
/// environment. Every access holds this lock, and the value never leaves the process.
static LAST_MEASURED: Mutex<Option<EnvKey>> = Mutex::new(None);

/// What "the locale in effect" is read from: the whole environment an agent's command would
/// get, in a canonical order, so the same environment always produces the same key.
///
/// Not just `LC_ALL`/`LC_CTYPE`/`LANG`: a GNU grep reads `POSIXLY_CORRECT`, and `PATH`
/// decides *which* grep the comparison ran. A verdict therefore stands for as long as this
/// key does — replacing a program the environment does not name (the owner's `grep` under
/// an unchanged `PATH`) is not noticed until the environment moves.
#[derive(PartialEq, Eq)]
struct EnvKey(Vec<(OsString, OsString)>);

impl EnvKey {
    /// The key of an environment pair list, canonically ordered, so an identical environment
    /// delivered in a different order keeps the verdict it was measured for.
    fn new(env: &[(OsString, OsString)]) -> Self {
        let mut pairs = env.to_vec();
        pairs.sort();
        EnvKey(pairs)
    }
}

/// Whether the fast path may stand for a member needing `needs` on `platform`.
///
/// `Ok(())` only when every need was established for the environment in effect; the `Err`
/// is the unmet capability's word, recorded as the demotion's reason. On Windows there is
/// no real search: the fast path stands and no battery is ever run, a behaviour drivable
/// from any host because the platform arrives as the caller's own value.
pub(super) fn serve_allowed(needs: &[Need], platform: ShellPlatform) -> Result<(), &'static str> {
    if platform == ShellPlatform::Windows {
        return Ok(());
    }
    // The battery's rows run through the production serve path: they must not be gated by
    // the verdict being measured (see [`Measuring`]).
    if MEASURING.with(std::cell::Cell::get) {
        return Ok(());
    }
    let verdict = established();
    needs
        .iter()
        .find(|need| !verdict.covers(**need))
        .map_or(Ok(()), |need| Err(need.capability()))
}

/// Re-measure the verdict when the environment an agent's command runs under differs from
/// the one the published verdict was measured for.
///
/// Blocking — the battery runs the real search — so callers hand it to a blocking pool; the
/// environment reader is the trigger. An environment already measured returns `true` without
/// measuring, the common case; nothing read from the environment is ever logged.
///
/// The verdict for the previous environment is dropped before the new one is measured, so
/// the two never mix: while the battery runs the fast path is closed and the real search
/// answers. A run that produced no verdict claims nothing — no key is kept and the gate
/// stays closed — so a transient failure is measured again by the next refresh rather than
/// closing the fast path for the life of the process; a shutdown starts no measurement and
/// stops one in flight at its next row boundary.
///
/// Returns whether a verdict for the environment in effect stands afterwards. `env` must be
/// the pairs a command gets right now — the same list [`invalidate_verdict_for`] was given,
/// or [`super::super::agent_env_pairs`] — so the verdict, the key it is filed under and what
/// a command runs under cannot drift apart.
pub(crate) fn refresh_if_environment_changed(env: &[(OsString, OsString)]) -> bool {
    // On Windows there is no real search, so there is nothing to measure; the reader checks
    // the platform too, so a failed read there does not ask the pool for this `false`.
    if cfg!(windows) {
        return false;
    }
    // Nothing new starts during a shutdown, and a running battery stops at its next row
    // boundary, so the reader's join is bounded by [`BATTERY_SPAWN_TIMEOUT`].
    if crate::shutdown::aborting() {
        return ESTABLISHED.load().is_some();
    }
    let key = EnvKey::new(env);
    if LAST_MEASURED.lock().unwrap_poison().as_ref() == Some(&key) {
        return true;
    }
    // Fail-closed for the window the battery takes; the reader has normally done this
    // already, and this covers any other way the environment moved.
    ESTABLISHED.store(None);
    record(key, measure(env))
}

/// Drop the verdict in effect when the environment commands are about to get is not the one
/// it was measured for. The reader calls this with the environment a fresh read will publish,
/// in the step *before* it publishes, so a command served in between still sees the old
/// environment — the one the verdict belongs to. Nothing is measured here.
pub(crate) fn invalidate_verdict_for(env: &[(OsString, OsString)]) {
    if cfg!(windows) {
        return;
    }
    let key = EnvKey::new(env);
    let mut last = LAST_MEASURED.lock().unwrap_poison();
    if last.as_ref() == Some(&key) {
        return;
    }
    ESTABLISHED.store(None);
    *last = None;
}

/// Record a measurement's outcome for `key`, and report whether a verdict is in effect
/// afterwards. A measured verdict is published with the key it was measured for; a run that
/// produced none clears both the key and the gate, so the next refresh measures again.
fn record(key: EnvKey, verdict: Option<Established>) -> bool {
    let mut last = LAST_MEASURED.lock().unwrap_poison();
    if let Some(verdict) = verdict {
        publish(verdict);
        *last = Some(key);
        true
    } else {
        ESTABLISHED.store(None);
        *last = None;
        false
    }
}

/// Install a verdict for the calling thread. The in-crate lanes that exercise the served path
/// are about the engine's equivalence proof against the locale those lanes pin, not about the
/// runtime gate, which has its own tests below; the per-thread override keeps a lane from
/// racing a sibling lane's publish.
#[cfg(test)]
pub(super) fn publish_for_test(verdict: (bool, bool, bool)) {
    TEST_VERDICT.with(|cell| {
        cell.set(Some(Established {
            plain: verdict.0,
            general: verdict.1,
            glob: verdict.2,
        }));
    });
}

/// The battery's tallies, and the verdict they fold into.
#[derive(Default)]
struct Battery {
    /// The capabilities this run proved. `None` when nothing usable was measured (an unusable
    /// probe, or a row that had to be compared and could not be): the run establishes nothing.
    verdict: Option<Established>,
    /// Rows attempted / compared / skipped / not measurable, as the record reports them.
    rows: usize,
    compared: usize,
    skipped: usize,
    /// Rows meant to be compared but not (one side did not run). Counted apart from skipped
    /// rows, because such a row is evidence of *nothing* and must not let a group be
    /// established from the rows that did run.
    unmeasured: usize,
    /// What the probe that decides whether a real search exists here found.
    real_search: Probe,
}

/// One capability's row tally. A capability is established only when at least one row was
/// compared and no row failed; a row that could not be compared is counted on the battery,
/// where it stops a verdict being folded from a partial run (see [`Battery::fold`]).
#[derive(Default)]
struct Group {
    compared: usize,
    failed: bool,
}

impl Battery {
    /// Fold a run's groups into its verdict: every capability from its own group, and nothing
    /// at all when a row that had to be compared could not be — such a run claims nothing, so
    /// the next read measures again rather than one transient failure keeping the real search.
    fn fold(&mut self, plain: &Group, general: &Group, glob: &Group) {
        if self.unmeasured > 0 {
            return;
        }
        self.verdict = Some(Established {
            plain: plain.established(),
            general: general.established(),
            glob: glob.established(),
        });
    }
}

/// What one row's comparison came to.
#[derive(Clone, Copy)]
enum RowOutcome {
    /// Both sides ran and agreed, or disagreed.
    Compared(bool),
    /// No parity question: the engine cannot serve this shape, or the row does not apply here.
    Skipped,
    /// The row had to be compared and could not be: one side did not run. Claims nothing about
    /// its capability — counted on the battery, which then folds no verdict at all.
    Unmeasured,
}

impl Group {
    /// Fold one row's outcome into the tallies.
    fn record(&mut self, battery: &mut Battery, outcome: RowOutcome) {
        battery.rows += 1;
        match outcome {
            RowOutcome::Compared(agreed) => {
                self.compared += 1;
                battery.compared += 1;
                self.failed |= !agreed;
            }
            RowOutcome::Unmeasured => battery.unmeasured += 1,
            RowOutcome::Skipped => battery.skipped += 1,
        }
    }

    const fn established(&self) -> bool {
        self.compared > 0 && !self.failed
    }
}

/// Measure every capability under `env`, with nothing allowed to panic out of the run: a
/// fixture that cannot be built, a panic inside, a probe that could not be run or a row that
/// had to be compared and could not yields no verdict at all, and only booleans and counts
/// are logged. `None` is "no verdict could be taken", as opposed to establishing nothing.
fn measure(env: &[(OsString, OsString)]) -> Option<Established> {
    let started = Instant::now();
    let Some(fixture) = Fixture::create() else {
        tracing::warn!("grep engine locale parity: fixture unavailable; fail-closed");
        return None;
    };
    let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_battery(&fixture, env)));
    let Ok(battery) = ran else {
        tracing::warn!("grep engine locale parity: battery panicked; fail-closed");
        return None;
    };
    let verdict = battery.verdict;
    tracing::debug!(
        established = verdict.is_some(),
        plain = verdict.is_some_and(|v| v.plain),
        general = verdict.is_some_and(|v| v.general),
        glob = verdict.is_some_and(|v| v.glob),
        rows = battery.rows,
        compared = battery.compared,
        skipped = battery.skipped,
        unmeasured = battery.unmeasured,
        real_search = ?battery.real_search,
        elapsed_ms = started.elapsed().as_millis(),
        "grep engine locale parity measured"
    );
    verdict
}

/// Run every row and fold the verdict. A host with no real search answers immediately — the
/// fast path is the only search, so it stands for every capability — while a probe that could
/// not be *run* establishes nothing, leaving the caller's fail-closed default. A shutdown
/// begun mid-run stops the rows at the next boundary, with the same effect.
fn run_battery(fixture: &Fixture, env: &[(OsString, OsString)]) -> Battery {
    let mut battery = Battery {
        real_search: real_search_exists(fixture, env),
        ..Default::default()
    };
    match battery.real_search {
        Probe::Absent => {
            battery.verdict = Some(Established::all());
            return battery;
        }
        Probe::Unknown => return battery,
        Probe::Present => {}
    }

    // A shutdown stops the battery at the next row boundary: the reader's join waits for the
    // whole run. The rows not run are not claimed, so the gate stays closed.
    let mut plain = Group::default();
    for row in plain_rows(fixture) {
        if crate::shutdown::aborting() {
            return battery;
        }
        plain.record(&mut battery, compare_grep_row(fixture, env, &row));
    }
    let mut general = Group::default();
    for row in general_rows(fixture) {
        if crate::shutdown::aborting() {
            return battery;
        }
        general.record(&mut battery, compare_grep_row(fixture, env, &row));
    }
    let mut glob = Group::default();
    for pattern in GLOB_PATTERNS {
        if crate::shutdown::aborting() {
            return battery;
        }
        glob.record(&mut battery, compare_glob_row(fixture, env, pattern));
    }
    // The trailing filter row runs a child too, so the shutdown check covers it like any row.
    if crate::shutdown::aborting() {
        return battery;
    }
    glob.record(
        &mut battery,
        compare_grep_row(fixture, env, &filter_row(fixture)),
    );

    // A row that could not be compared is evidence of nothing, and such a run is not a verdict
    // for this environment (see [`Battery::fold`]).
    battery.fold(&plain, &general, &glob);
    battery
}

/// One differential row: the verb-first argv both sides receive, plus whether the row's stdout
/// compares as a sorted record set (recursive walks only — the parallel walk is an approved one).
struct Row {
    argv: Vec<String>,
    sorted: bool,
}

impl Row {
    /// A row over `argv` (verb first), with absolute operand paths so both sides print the same
    /// `file:` prefixes.
    fn new(argv: &[&str], sorted: bool) -> Self {
        Row {
            argv: argv.iter().map(|word| (*word).to_string()).collect(),
            sorted,
        }
    }
}

/// The literal shapes: byte matching over the ascii, unicode, invalid-UTF-8 and binary fixtures,
/// every flag that adds no case or word boundary, one two-file row, and one clean recursive walk.
fn plain_rows(fixture: &Fixture) -> Vec<Row> {
    let a = fixture.abs("a.txt");
    let b = fixture.abs("b.txt");
    let uni = fixture.abs("uni.txt");
    let bad = fixture.abs("bad.txt");
    let bin = fixture.abs("bin.dat");
    let dir = fixture.abs("dir");
    vec![
        Row::new(&["grep", "needle", a.as_str()], false),
        Row::new(&["grep", "needle", b.as_str()], false),
        Row::new(&["grep", "needle", uni.as_str()], false),
        Row::new(&["grep", "needle", bad.as_str()], false),
        Row::new(&["grep", "needle", bin.as_str()], false),
        Row::new(&["grep", "-n", "needle", a.as_str()], false),
        Row::new(&["grep", "-c", "needle", a.as_str()], false),
        Row::new(&["grep", "-l", "needle", a.as_str(), b.as_str()], false),
        Row::new(&["grep", "-v", "needle", a.as_str()], false),
        Row::new(&["grep", "-x", "needle", a.as_str()], false),
        Row::new(&["grep", "-o", "needle", a.as_str()], false),
        Row::new(&["grep", "-m1", "needle", a.as_str(), b.as_str()], false),
        Row::new(&["grep", "-h", "needle", a.as_str(), b.as_str()], false),
        Row::new(&["grep", "-H", "needle", a.as_str()], false),
        Row::new(&["grep", "-F", "a.b", a.as_str(), b.as_str()], false),
        Row::new(&["grep", "-e", "needle", "-e", "alpha", a.as_str()], false),
        Row::new(&["grep", "needle", a.as_str(), b.as_str()], false),
        Row::new(&["grep", "-r", "needle", dir.as_str()], true),
    ]
}

/// The shapes a locale reaches into: Unicode case folding over tricky code points, POSIX classes
/// against non-ASCII data, `.` over multibyte data, an accented bracket set, a word boundary, and
/// `-i` over the invalid-UTF-8 and binary fixtures.
fn general_rows(fixture: &Fixture) -> Vec<Row> {
    let a = fixture.abs("a.txt");
    let uni = fixture.abs("uni.txt");
    let bad = fixture.abs("bad.txt");
    let bin = fixture.abs("bin.dat");
    vec![
        Row::new(&["grep", "-i", "k", uni.as_str()], false),
        Row::new(&["grep", "-i", "i", uni.as_str()], false),
        Row::new(&["grep", "-i", "s", uni.as_str()], false),
        Row::new(&["grep", "-i", "ß", uni.as_str()], false),
        Row::new(&["grep", "-i", "µ", uni.as_str()], false),
        Row::new(&["grep", "-i", "σ", uni.as_str()], false),
        Row::new(&["grep", "[[:alpha:]]", uni.as_str()], false),
        Row::new(&["grep", "[[:lower:]]", uni.as_str()], false),
        Row::new(&["grep", "[[:upper:]]", uni.as_str()], false),
        Row::new(&["grep", "[[:digit:]]", uni.as_str()], false),
        Row::new(&["grep", "a.c", uni.as_str()], false),
        Row::new(&["grep", "[à]", uni.as_str()], false),
        Row::new(&["grep", "-w", "needle", a.as_str()], false),
        Row::new(&["grep", "-i", "needle", bad.as_str()], false),
        Row::new(&["grep", "-i", "needle", bin.as_str()], false),
    ]
}

/// The glob operands the glob rows expand. The names tree holds an ASCII-uppercase name, an
/// accented pair and a plain name, so collation and character classes can disagree.
const GLOB_PATTERNS: &[&str] = &["*.txt", "?.txt", "[à]*.txt", "[a-z]*.txt", "[!a]*.txt"];

/// The filter row: `--include`'s `fnmatch` is read by the engine from a process that never calls
/// `setlocale`, and by the real search under the locale in effect.
fn filter_row(fixture: &Fixture) -> Row {
    Row::new(
        &[
            "grep",
            "-rn",
            "--include=*.txt",
            "needle",
            fixture.abs("dir").as_str(),
        ],
        true,
    )
}

/// Compare one grep row end to end: same argv, same environment, engine in-process against
/// the real search. A shape the engine cannot serve is no parity question
/// ([`RowOutcome::Skipped`]), while a real search that did not run says nothing about the row
/// ([`RowOutcome::Unmeasured`]), so the capability stays fail-closed.
fn compare_grep_row(fixture: &Fixture, env: &[(OsString, OsString)], row: &Row) -> RowOutcome {
    let Some(spec) = spec_for(&segment_text(&row.argv), fixture, ShellPlatform::Unix) else {
        return RowOutcome::Skipped;
    };
    let Some((engine_out, engine_code)) = engine_run(&spec) else {
        return RowOutcome::Skipped;
    };
    let Child::Ran(real_out, real_code) = real_search(&row.argv, env, &fixture.root) else {
        return RowOutcome::Unmeasured;
    };
    RowOutcome::Compared(equivalent(
        &engine_out,
        engine_code,
        &real_out,
        real_code,
        row.sorted,
    ))
}

/// Build the spec the production parent path builds for `segment`, so the comparison
/// exercises the served shape rather than a hand-built one; `None` when that path refuses.
/// One deliberate deviation: `allow_single` is `true` (the Windows production value), which
/// bypasses the unix single-file perf gate — a single-file lookup is exactly a shape the
/// engine must be held to. The call suspends the gate for its duration (see [`Measuring`]).
fn spec_for(segment: &str, fixture: &Fixture, platform: ShellPlatform) -> Option<EngineSpec> {
    let _measuring = Measuring::enter();
    serve_one_grep(
        segment,
        "grep",
        &fixture.root,
        &fixture.home,
        platform,
        true,
        PipelineCtx {
            piped: false,
            stdin_fed: false,
            marker_ok: true,
            // The battery builds one member at a time, never a line that exported something.
            env_changed: false,
        },
    )
    .ok()
    .map(|(spec, _redirects)| spec)
}

/// Compare [`expand_glob`] with the shell's own expansion of the same operand: same members,
/// same order. As in [`compare_grep_row`], a shape the engine cannot serve is no parity
/// question and a shell that did not run is nothing to measure.
fn compare_glob_row(fixture: &Fixture, env: &[(OsString, OsString)], pattern: &str) -> RowOutcome {
    let Some(engine) = expand_glob(pattern, &fixture.names, ShellPlatform::Unix).ok() else {
        return RowOutcome::Skipped;
    };
    let Some(shell) = shell_glob(pattern, fixture, env) else {
        return RowOutcome::Unmeasured;
    };
    RowOutcome::Compared(engine == shell)
}

/// The shell's own expansion of one glob operand, one name per line:
/// `sh -c 'printf "%s\n" <pattern>'` under the agent's environment, cwd = the names tree.
/// `None` when `sh` could not be run at all, which measures nothing.
fn shell_glob(
    pattern: &str,
    fixture: &Fixture,
    env: &[(OsString, OsString)],
) -> Option<Vec<String>> {
    let script = format!("printf \"%s\\n\" {pattern}");
    let mut cmd = agent_command("sh", env, &fixture.names);
    cmd.arg("-c").arg(&script);
    let Child::Ran(stdout, _code) = bounded_output(&mut cmd) else {
        return None;
    };
    Some(
        String::from_utf8_lossy(&stdout)
            .lines()
            .map(str::to_string)
            .collect(),
    )
}

/// Run the real search by name with the agent's environment applied and the fixture root as cwd.
fn real_search(argv: &[String], env: &[(OsString, OsString)], cwd: &Path) -> Child {
    let Some((verb, operands)) = argv.split_first() else {
        // No program name at all: nothing ran, so nothing was measured.
        return Child::Unusable;
    };
    let mut cmd = agent_command(verb, env, cwd);
    cmd.args(operands);
    bounded_output(&mut cmd)
}

/// A command with the agent's environment applied whole (nothing inherited), run in `cwd`,
/// with stdin and stderr taken away: the battery runs the owner's own programs unattended in
/// the background, so nothing may read the daemon's stdin or spill into its stderr. Stdout is
/// the row's capture ([`bounded_output`]).
///
/// The program is named by the path that environment's search list resolves it to
/// ([`crate::tools::shell::program_command`]), which is why `cwd` sits here: an empty or
/// relative search entry resolves against it and an unresolvable name stays bare, leaving
/// [`Child::Missing`] unchanged. The window guarantee `tools::shell::tree` documents holds too.
/// The battery never runs on Windows.
fn agent_command(program: &str, env: &[(OsString, OsString)], cwd: &Path) -> Command {
    let mut cmd = crate::tools::shell::program_command(program, env, Some(cwd));
    cmd.stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    cmd
}

/// One battery child's result: its stdout and exit code, or why it produced neither. The two
/// failure arms are what [`real_search_exists`] needs — "not installed here" is not "could not
/// be started".
enum Child {
    /// Ran to completion inside the deadline.
    Ran(Vec<u8>, i32),
    /// Nothing in this environment's search path resolves to it: the program is not installed.
    Missing,
    /// No usable result: it could not be started for another reason, failed to be waited on, or
    /// outlived [`BATTERY_SPAWN_TIMEOUT`] and was killed.
    Unusable,
}

/// Run one battery child to completion under a deadline, and hand back what it wrote.
///
/// The deadline keeps the battery — and the environment reader that hands the measurement to a
/// blocking pool and waits for it — from depending on a program the owner's `PATH` resolves: a
/// `grep` that hangs would otherwise wedge a blocking thread and stop every later read.
///
/// The child's stdout is a file, not a pipe, which is what makes the deadline cover the whole
/// row: a pipe must be read to EOF, and an EOF a stray holder of the write end never produces
/// would block this thread for good. A killed child is handed to [`reap`]. The capture is
/// capped at [`BATTERY_STDOUT_CAP`], and a child already past it is killed rather than waited
/// for. (Its stdout is still not a terminal.)
fn bounded_output(cmd: &mut Command) -> Child {
    let Ok(sink) = tempfile::tempfile_in(crate::temp::shell_tmpdir()) else {
        return Child::Unusable;
    };
    // A handle of our own to read through: the sink itself goes to the child.
    let Ok(mut captured) = sink.try_clone() else {
        return Child::Unusable;
    };
    let mut child = match cmd.stdout(std::process::Stdio::from(sink)).spawn() {
        Ok(child) => child,
        // "Not on this system" is a finding; any other spawn failure is not evidence of absence.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Child::Missing,
        Err(_) => return Child::Unusable,
    };
    let deadline = std::time::Instant::now() + BATTERY_SPAWN_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            // A wait error is as unusable as a failure to start.
            Err(_) => return Child::Unusable,
            Ok(None) if std::time::Instant::now() >= deadline => {
                reap(&mut child);
                return Child::Unusable;
            }
            // Nothing a row needs comes near the cap, so a child already past it is a runaway:
            // killed, and the row is unusable rather than captured.
            Ok(None)
                if captured
                    .metadata()
                    .is_ok_and(|file| file.len() > BATTERY_STDOUT_CAP) =>
            {
                reap(&mut child);
                return Child::Unusable;
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(5)),
        }
    };
    // A clone shares the file's offset with the writer, so rewind before reading.
    if captured.seek(std::io::SeekFrom::Start(0)).is_err() {
        return Child::Unusable;
    }
    let mut out = Vec::new();
    if captured
        .take(BATTERY_STDOUT_CAP)
        .read_to_end(&mut out)
        .is_err()
    {
        return Child::Unusable;
    }
    Child::Ran(out, status.code().unwrap_or(-1))
}

/// Kill a battery child and collect it, under the same bound as the row it belongs to.
///
/// The wait is bounded for the row's reason: a child that survives SIGKILL must not park the
/// blocking thread the environment reader joins. One still running at the bound is left to
/// the OS rather than waited for.
fn reap(child: &mut std::process::Child) {
    let _ = child.kill();
    let deadline = std::time::Instant::now() + BATTERY_SPAWN_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(_)) | Err(_) => return,
            Ok(None) if std::time::Instant::now() >= deadline => return,
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(5)),
        }
    }
}

/// Whether a real search exists here at all. Three-way, because "nothing on this system to
/// compare against" is a verdict of its own — the fast path is then the only search and
/// stands — while a probe that could not be *run* must leave the verdict fail-closed.
fn real_search_exists(fixture: &Fixture, env: &[(OsString, OsString)]) -> Probe {
    let argv = [
        "grep".to_string(),
        "needle".to_string(),
        fixture.abs("a.txt"),
    ];
    match real_search(&argv, env, &fixture.root) {
        Child::Ran(..) => Probe::Present,
        Child::Missing => Probe::Absent,
        Child::Unusable => Probe::Unknown,
    }
}

/// What [`real_search_exists`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Probe {
    /// The real search ran: the battery compares against it.
    Present,
    /// Nothing here resolves to a real search: the fast path is the only search and stands.
    Absent,
    /// The probe itself could not be run. Not evidence of absence: nothing is established.
    #[default]
    Unknown,
}

/// Run one spec in-process as the existing lanes do: the production matcher builder, a buffer
/// sink standing in for the capture pipe, and empty stdin. `None` when the matcher cannot build.
fn engine_run(spec: &EngineSpec) -> Option<(Vec<u8>, i32)> {
    let matcher = build_matcher(&spec.patterns, spec.mode, &spec.flags).ok()?;
    let mut out = Output::new(OutputSink::Buffer(Vec::new()), StdoutDest::of(spec, true));
    let consumed = std::cell::Cell::new(false);
    let code = serve_into(spec, &matcher, &mut out, io::empty(), &consumed);
    let (bytes, _err) = out.take_stdio();
    Some((bytes, code))
}

/// Byte-exact stdout plus an equal exit code, relaxed to sorted record sets for recursive rows.
fn equivalent(engine: &[u8], engine_code: i32, real: &[u8], real_code: i32, sorted: bool) -> bool {
    if engine_code != real_code {
        return false;
    }
    if sorted {
        sorted_lines(engine) == sorted_lines(real)
    } else {
        engine == real
    }
}

/// The non-empty records of a search result, byte-sorted.
fn sorted_lines(bytes: &[u8]) -> Vec<&[u8]> {
    let mut records: Vec<&[u8]> = bytes
        .split(|&b| b == b'\n')
        .filter(|r| !r.is_empty())
        .collect();
    records.sort_unstable();
    records
}

/// The shell segment the parent-side reader is handed for a row: the same argv words,
/// shell-quoted, so the tokenizer sees the spellings the real search receives.
fn segment_text(argv: &[String]) -> String {
    argv.iter()
        .map(|word| shell_quote(word))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The byte-exact fixture the battery searches: a scratch tree under the shell temp root.
struct Fixture {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    home: PathBuf,
    names: PathBuf,
}

impl Fixture {
    /// Build the fixture tree inside [`crate::temp::shell_tmpdir`]. `None` when the directory
    /// or any fixture file cannot be created — the caller stays fail-closed.
    fn create() -> Option<Self> {
        let tmp = tempfile::tempdir_in(crate::temp::shell_tmpdir()).ok()?;
        let root = tmp.path().to_path_buf();
        std::fs::create_dir_all(root.join("dir/sub")).ok()?;
        std::fs::create_dir_all(root.join("names")).ok()?;
        std::fs::create_dir_all(root.join("home")).ok()?;
        // Byte-exact: the rows depend on these code points and invalid bytes, not on any
        // encoding a writer might apply. The `dir/` tree has no hidden entries, no ignore files
        // and no binary files, so the rg-default walk exclusions cannot cause a difference.
        let files: &[(&str, &[u8])] = &[
            ("a.txt", b"alpha\nbeta\nneedle\n"),
            ("b.txt", b"needle\nother\n"),
            (
                "uni.txt",
                "café\nCAFÉ\nk\nK\n\u{212a}\nstraßen\nSTRASSE\nistanbul\n\u{0130}stanbul\n"
                    .as_bytes(),
            ),
            ("bad.txt", b"bad\x80\xff needle\ntail\n"),
            ("bin.dat", b"\x00\x01 needle\n"),
            ("dir/two.txt", b"needle\n"),
            ("dir/sub/one.txt", b"other\nneedle\n"),
            ("names/a.txt", b"x\n"),
            ("names/B.txt", b"x\n"),
            ("names/\u{00c3}.txt", b"x\n"),
            ("names/\u{00e0}.txt", b"x\n"),
            ("names/\u{00e9}.txt", b"x\n"),
        ];
        for (name, bytes) in files {
            std::fs::write(root.join(name), bytes).ok()?;
        }
        Some(Fixture {
            names: root.join("names"),
            home: root.join("home"),
            root,
            _tmp: tmp,
        })
    }

    /// An operand spelled absolutely (so both sides print the same `file:` prefix).
    fn abs(&self, rel: &str) -> String {
        self.root.join(rel).to_string_lossy().into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Fallback, PipelineCtx, grep_tokenize, parse_grep_words, serve_one_grep};
    use super::*;

    /// Parse a grep segment as the serve path does, so a `needs` row reads the same [`ParsedGrep`].
    fn parsed(segment: &str) -> ParsedGrep {
        let mut words = grep_tokenize(segment, ShellPlatform::Unix).expect("tokenize");
        words.remove(0);
        parse_grep_words(&words, "grep", ShellPlatform::Unix).expect("parse")
    }

    /// Each row outcome lands in its own tally, and a run that could not compare a row it had
    /// to folds no verdict at all — so the next read measures again.
    #[test]
    fn a_run_with_a_row_it_could_not_compare_folds_no_verdict() {
        let mut battery = Battery::default();
        let mut plain = Group::default();
        plain.record(&mut battery, RowOutcome::Compared(true));
        plain.record(&mut battery, RowOutcome::Skipped);
        assert_eq!(
            (
                battery.rows,
                battery.compared,
                battery.skipped,
                battery.unmeasured
            ),
            (2, 1, 1, 0),
            "each outcome is counted where it belongs"
        );
        battery.fold(&plain, &Group::default(), &Group::default());
        assert!(
            battery.verdict.is_some_and(|v| v.plain),
            "a run that compared every row it had to folds its verdict"
        );

        let mut battery = Battery::default();
        let mut plain = Group::default();
        plain.record(&mut battery, RowOutcome::Compared(true));
        plain.record(&mut battery, RowOutcome::Unmeasured);
        assert_eq!(battery.unmeasured, 1, "an unmeasurable row is counted");
        battery.fold(&plain, &Group::default(), &Group::default());
        assert!(
            battery.verdict.is_none(),
            "one unmeasurable row is not a verdict for the environment"
        );
    }

    /// The verdict and the key it was measured for move together: a verdict is in effect only
    /// while its environment is, and a measurement that produced none claims nothing — so a
    /// transient failure cannot leave the gate closed for the life of the process.
    ///
    /// Driven directly, because a real measurement spawns the owner's own programs.
    /// `#[serial]`: the verdict and its key are process-global.
    #[cfg(unix)]
    #[serial_test::serial(shell_env)]
    #[test]
    fn a_verdict_stands_only_for_the_environment_it_was_measured_for() {
        let in_effect = || EnvKey::new(&crate::tools::shell::agent_env_pairs());
        let other = [(OsString::from("ONE"), OsString::from("two"))];
        // The gate itself, not `established()`: the latter prefers the test-only per-thread
        // override, and this lane is about the process-global verdict.
        let gate = || ESTABLISHED.load().as_deref().copied().unwrap_or_default();

        // A measured environment is in effect; looking again at the *same* one measures nothing.
        assert!(record(in_effect(), Some(Established::all())));
        assert_eq!(gate(), Established::all());
        assert!(
            refresh_if_environment_changed(&crate::tools::shell::agent_env_pairs()),
            "an environment measured with a verdict is a no-op"
        );

        // A run that produced no verdict claims nothing: the gate closes and no key is left.
        assert!(!record(in_effect(), None));
        assert_eq!(gate(), Established::default());
        assert!(LAST_MEASURED.lock().unwrap_poison().is_none());

        // An environment that is not the measured one drops the verdict and leaves no key.
        assert!(record(EnvKey::new(&other), Some(Established::all())));
        assert_eq!(gate(), Established::all());
        invalidate_verdict_for(&crate::tools::shell::agent_env_pairs());
        assert_eq!(gate(), Established::default());
        assert!(LAST_MEASURED.lock().unwrap_poison().is_none());

        // An environment that *is* the measured one keeps its verdict.
        assert!(record(EnvKey::new(&other), Some(Established::all())));
        invalidate_verdict_for(&other);
        assert_eq!(gate(), Established::all());
        assert!(LAST_MEASURED.lock().unwrap_poison().as_ref() == Some(&EnvKey::new(&other)));

        // Leave the process as this lane found it.
        ESTABLISHED.store(None);
        *LAST_MEASURED.lock().unwrap_poison() = None;
    }

    /// `needs` in one table: byte literals keep the fast path, anything a locale reaches into
    /// is general, and a glob operand or a filter adds the glob dimension.
    #[test]
    fn needs_follows_the_member_shape() {
        let rows: &[(&str, &[Need])] = &[
            ("grep needle a.txt", &[Need::Plain]),
            ("grep -n needle a.txt", &[Need::Plain]),
            ("grep -F needle a.txt", &[Need::Plain]),
            ("grep -x needle a.txt", &[Need::Plain]),
            ("grep -i needle a.txt", &[Need::General]),
            ("grep -w needle a.txt", &[Need::General]),
            ("grep 'a.c' a.txt", &[Need::General]),
            ("grep -F 'a.b' a.txt", &[Need::Plain]),
            (r"grep 'a\|b' a.txt", &[Need::General]),
            ("grep needle *.txt", &[Need::Plain, Need::Glob]),
            (
                "grep needle --include=*.txt a.txt",
                &[Need::Plain, Need::Glob],
            ),
            (
                "grep needle --exclude-dir=sub a.txt",
                &[Need::Plain, Need::Glob],
            ),
            ("grep -i needle '*.txt'", &[Need::General]),
        ];
        for (segment, expected) in rows {
            assert_eq!(
                needs(&parsed(segment), ShellPlatform::Unix),
                *expected,
                "{segment}"
            );
        }
    }

    /// The gate demotes every capability that was not established, naming the unmet one.
    #[test]
    fn the_gate_demotes_what_was_not_established() {
        publish_for_test((false, false, false));
        assert_eq!(
            serve_allowed(&[Need::Plain], ShellPlatform::Unix),
            Err("literal matching")
        );
        assert_eq!(
            serve_allowed(&[Need::General], ShellPlatform::Unix),
            Err("pattern matching")
        );
        assert_eq!(
            serve_allowed(&[Need::Glob], ShellPlatform::Unix),
            Err("glob expansion")
        );

        publish_for_test((true, false, true));
        assert_eq!(serve_allowed(&[Need::Plain], ShellPlatform::Unix), Ok(()));
        assert_eq!(serve_allowed(&[Need::Glob], ShellPlatform::Unix), Ok(()));
        assert_eq!(
            serve_allowed(&[Need::General], ShellPlatform::Unix),
            Err("pattern matching")
        );

        publish_for_test((true, true, true));
        assert_eq!(
            serve_allowed(&[Need::Plain, Need::Glob], ShellPlatform::Unix),
            Ok(())
        );
    }

    /// A host with no real search: the battery establishes every capability rather than leaving
    /// the fail-closed default a probe that could not be *run* leaves in place.
    #[cfg(unix)]
    #[test]
    fn a_host_without_a_real_search_establishes_every_capability() {
        let fixture = Fixture::create().expect("fixture");
        let empty = tempfile::tempdir().expect("empty dir");
        let env = vec![(
            OsString::from("PATH"),
            empty.path().as_os_str().to_os_string(),
        )];
        let battery = run_battery(&fixture, &env);
        assert_eq!(battery.real_search, Probe::Absent, "no search in that PATH");
        assert_eq!(
            battery.verdict,
            Some(Established::all()),
            "the fast path is the only search here"
        );
        assert_eq!(battery.rows, 0, "nothing was compared");
    }

    /// The gate never fires on Windows: the fast path is the only search and stands however the
    /// verdict stands. The platform is a value, so this is drivable from this host.
    #[test]
    fn a_windows_member_is_served_however_the_verdict_stands() {
        publish_for_test((false, false, false));
        let fixture = Fixture::create().expect("fixture");
        let segment = "grep -n needle a.txt";
        let serve = |platform| {
            serve_one_grep(
                segment,
                "grep",
                &fixture.root,
                &fixture.home,
                platform,
                true,
                PipelineCtx {
                    piped: false,
                    stdin_fed: false,
                    marker_ok: true,
                    env_changed: false,
                },
            )
        };
        assert!(
            serve(ShellPlatform::Windows).is_ok(),
            "a Windows serve never consults the verdict"
        );
        assert!(
            matches!(
                serve(ShellPlatform::Unix),
                Err(Fallback::NotEstablished("literal matching"))
            ),
            "the same member is demoted on unix with nothing established"
        );
    }
}
