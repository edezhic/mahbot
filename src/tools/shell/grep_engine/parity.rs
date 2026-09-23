//! The runtime locale-parity gate for the grep engine's fast served path.
//!
//! The engine's matching model is its own — Rust regex over UTF-8, with the
//! C.UTF-8 deltas the module header lists (Unicode simple case folding, ASCII
//! POSIX classes, invalid-UTF-8 lines treated as non-matches) — while the search
//! it stands in for is a locale-aware `grep` running in the environment the
//! owner's own terminal gives the agent's command. That environment is the
//! owner's ([`super::super::agent_env_pairs`]), so its locale is unknown to the
//! product: the engine may no longer assume the `LC_ALL=C.UTF-8` it was written
//! against.
//!
//! A serve decision therefore cannot rest on the engine's own say-so. This
//! module publishes a coarse, runtime-measured verdict ([`Established`], read
//! through [`serve_allowed`]) and the parent-side serve path consults it on every
//! member: the fast path may stand only where the product established, under the
//! locale actually in effect, that the engine agrees with the real search for
//! that member's shape ([`Need`]). Where nothing was established the real search
//! runs instead; where there is no real search at all — Windows, or a unix
//! environment whose search path resolves none — the fast path is the only search
//! and stands.
//!
//! The verdict comes from a bounded differential battery ([`measure`]) over a
//! byte-exact fixture: the same rows run through the production parent path
//! in-process and through the real `grep`/`sh` under the agent's environment, and
//! a capability is established only when every compared row agreed (stdout bytes
//! and exit code; sorted record sets for recursive walks). The battery logs
//! booleans, counts and its probe's own finding only — never a locale name, never
//! anything else read from the environment.
//!
//! Its rows deliberately include the engine's own approved deltas — the Unicode
//! simple case folding that makes `-i k` match KELVIN, the ASCII-only POSIX
//! classes, the C.UTF-8 reading of invalid UTF-8 — so [`Need::General`] is
//! *expected* to stay unestablished on a host whose `grep` folds case by
//! lowercase comparison, and a `C` locale is expected to unestablish
//! [`Need::Plain`] as well. That is the point: the fast path kept for a shape is
//! the shape this host's own search was measured to agree with, and everything
//! else runs the real search.
//!
//! A verdict belongs to the environment it was measured under: the refresher
//! ([`refresh_if_environment_changed`]) drops the published one before measuring
//! a changed environment — normally in the reader's own step, before the new
//! environment is in place — so the window a battery takes is served fail-closed
//! — by the real search — rather than under a verdict for an environment no
//! command is running in any more.

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

/// Deadline on one battery child.
///
/// Generous against the milliseconds a real `grep` or `sh` takes on the fixture,
/// and small enough that an owner-`PATH` program which never finishes cannot
/// wedge the measurement, the blocking thread it runs on, or the environment
/// reader that waits for it.
const BATTERY_SPAWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Cap on one battery child's stdout, both what is read back and what a child may
/// write before it is killed.
///
/// A row's output is a few lines over a handful of tiny fixture files, so this is
/// far above anything a comparison needs: it bounds what a runaway program on the
/// owner's `PATH` can cost the reader's private temp root and its memory.
const BATTERY_STDOUT_CAP: u64 = 1024 * 1024;

// ── Capabilities ──────────────────────────────────────────────────────────

/// The one thing a member's shape is judged by, deliberately coarse.
///
/// A served search agrees with the real one either because its matching cannot
/// involve a locale at all ([`Need::Plain`]) or because it can and the engine's
/// model of it was proven under the locale in effect ([`Need::General`]); glob
/// operands and filters add a separate dimension ([`Need::Glob`]) because the
/// engine globs with libc `fnmatch` from a process that never calls `setlocale`,
/// while the shell and grep glob under the locale they were handed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Need {
    /// A literal, case-sensitive byte search: no `-i`, no `-w`, and no regex
    /// metacharacter or backslash in any raw pattern (`-F` patterns are literal
    /// whatever they contain). This is the one shape whose *engine-side* matching
    /// has no locale input at all, so it is the shape worth measuring instead of
    /// assuming: whether the real search agrees byte for byte is what the battery
    /// establishes, and it demonstrably does not under every locale (a `C` locale
    /// reads an invalid-UTF-8 line the engine's model treats as a non-match).
    Plain,
    /// Everything else — case-insensitive matching, POSIX classes, `.`,
    /// brackets, `-w`: the engine's model of it differs from a locale-aware
    /// grep's, so it needs a proof under the locale in effect.
    General,
    /// The member's operands carry an unquoted glob, or it carries
    /// `--include`/`--exclude`/`--exclude-dir` filters: `fnmatch` is
    /// locale-sensitive.
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

/// The capabilities one member's shape needs: always exactly one of
/// [`Need::Plain`]/[`Need::General`], plus [`Need::Glob`] when its operands or
/// filters carry a glob surface.
///
/// Fail-closed by construction: anything the reading below does not positively
/// recognise as a byte-literal shape is [`Need::General`]. The platform is the
/// caller's own value, so the shape reading cannot silently disagree with the
/// serve decision taken for the same member.
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

/// Whether the member's matching cannot depend on either search's locale.
///
/// True when every raw pattern is a literal in every dialect — `-F` makes that
/// true by definition, and outside it a pattern containing a metacharacter or a
/// backslash is not read as one — and no flag puts case or word boundaries
/// between the pattern and the bytes. What the *engine* does with such a pattern
/// has no locale input; what the real search does with it is what the battery
/// measures before the fast path may stand.
fn is_byte_literal(parsed: &ParsedGrep) -> bool {
    if parsed.flags.i || parsed.flags.w {
        return false;
    }
    // `-F` patterns are fixed strings whatever they contain; the engine
    // escapes them into a literal matcher, and the real search compares bytes.
    parsed.mode == MatchMode::Fixed
        || parsed.patterns.iter().all(|pattern| {
            !pattern.contains([
                '.', '^', '$', '*', '+', '?', '(', ')', '[', ']', '{', '}', '|', '\\',
            ])
        })
}

/// Whether the member carries a glob surface the shell would expand under its
/// own locale: an unquoted glob in an operand token, or an
/// `--include`/`--exclude`/`--exclude-dir` filter.
fn has_glob_surface(parsed: &ParsedGrep, platform: ShellPlatform) -> bool {
    !parsed.filters.is_empty()
        || !parsed.exclude_dir.is_empty()
        || parsed
            .operand_tokens
            .iter()
            .any(|tok| has_unquoted_glob(tok, platform))
}

// ── The published verdict ─────────────────────────────────────────────────

/// What the battery established for the locale a measurement was taken under,
/// one flag per [`Need`]. `Default` is all false: fail-closed until measured.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Established {
    plain: bool,
    general: bool,
    glob: bool,
}

impl Established {
    /// Every capability established.
    const fn all() -> Self {
        Established {
            plain: true,
            general: true,
            glob: true,
        }
    }

    /// Whether this verdict covers `need`.
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

// A per-thread verdict override, installed by test lanes.
//
// A lane that publishes a verdict is describing its own runs only: a
// process-global publish would race the sibling lanes that run concurrently in
// the same test binary, so the override is per-thread.
#[cfg(test)]
thread_local! {
    static TEST_VERDICT: std::cell::Cell<Option<Established>> =
        const { std::cell::Cell::new(None) };
}

/// The verdict in effect: a test thread's override when one is installed, else
/// the measured global (fail-closed until a measurement publishes one).
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

// Whether the calling thread is inside the battery's own row building.
//
// The battery builds each row's spec through the production parent path, which
// consults this same gate: without the flag, a fail-closed verdict would demote
// the very rows the verdict is measured from. It is per-thread, so no other
// thread's serve decision is ever affected, and it is scoped to the one call that
// reads the gate ([`spec_for`]).
thread_local! {
    static MEASURING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Suspends the gate for its lifetime (see [`MEASURING`]); the drop covers a
/// parent-path panic, so a thread can never be left gated open.
struct Measuring;

impl Measuring {
    /// Suspend the gate on this thread until the returned guard drops.
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

/// The environment the verdict in effect was measured for: set by [`record`]
/// together with the verdict, cleared by [`invalidate_verdict_for`] (before a
/// command can see the new environment) and by [`record`] when a run measured
/// nothing. Read by the changed-check of [`refresh_if_environment_changed`] and by
/// [`invalidate_verdict_for`], which must know whether the verdict in effect
/// belongs to the environment being replaced. Every access holds this lock, and
/// the value never leaves the process.
static LAST_MEASURED: Mutex<Option<EnvKey>> = Mutex::new(None);

/// What "the locale in effect" is read from: the whole environment an agent's
/// command would get, in a canonical order, so that the same environment always
/// produces the same key.
///
/// Not just `LC_ALL`/`LC_CTYPE`/`LANG`: what the real search does depends on more
/// of the environment than its locale (a GNU grep reads `POSIXLY_CORRECT`, and
/// `PATH` decides *which* grep the comparison even ran), and keying on the whole
/// environment is both simpler to reason about and impossible to get subtly
/// wrong. It costs one comparison per read — the environment only changes when a
/// read returns something new — and it never holds a value anywhere but in
/// memory.
///
/// A verdict therefore stands for as long as this key does: replacing a program
/// the environment does not name (the owner's `grep` binary under an unchanged
/// `PATH`) is not noticed until the environment itself moves. The bar is the
/// environment his commands get, which is what the key holds.
#[derive(PartialEq, Eq)]
struct EnvKey(Vec<(OsString, OsString)>);

impl EnvKey {
    /// The key of an environment pair list, canonically ordered: an environment
    /// is a map, so an identical one delivered in a different order is the same
    /// key and keeps the verdict it was measured for.
    fn new(env: &[(OsString, OsString)]) -> Self {
        let mut pairs = env.to_vec();
        pairs.sort();
        EnvKey(pairs)
    }
}

/// Whether the fast path may stand for a member needing `needs` on `platform`.
///
/// `Ok(())` only when every need was established for the environment in effect;
/// the `Err` is the unmet capability's word, which the parent-side serve path
/// records as the demotion's reason. On Windows there is no real search to run:
/// the fast path is the only search and stands, and no battery is ever run there.
/// The platform arrives as the caller's own value — the same one the member was
/// read under — so the Windows behaviour is drivable (and tested) from any host.
pub(super) fn serve_allowed(needs: &[Need], platform: ShellPlatform) -> Result<(), &'static str> {
    if platform == ShellPlatform::Windows {
        return Ok(());
    }
    // The battery's own row building runs through the production serve path, so
    // it must not be gated by the verdict it is measuring (see [`Measuring`]).
    if MEASURING.with(std::cell::Cell::get) {
        return Ok(());
    }
    let verdict = established();
    needs
        .iter()
        .find(|need| !verdict.covers(**need))
        .map_or(Ok(()), |need| Err(need.capability()))
}

/// Re-measure the verdict when the environment an agent's command runs under
/// differs from the one the published verdict was measured for.
///
/// Blocking — the battery runs the real search — so callers hand it to a
/// blocking pool. The environment reader is the right trigger: it is what changes
/// the environment an agent's commands get, and it calls this right after
/// publishing a read (and after a failed read, where it measures the fallback if
/// that has not been measured yet). An environment already measured returns
/// `true` without measuring, which is the common case. Nothing read from the
/// environment is ever logged.
///
/// The verdict for the previous environment is dropped before the new one is
/// measured ([`invalidate_verdict_for`] has normally done it already, in the
/// reader's own step), so the two never mix: while the battery runs, the fast
/// path is closed and the real search answers instead. A run that produced no
/// verdict claims nothing — no key is kept and the gate stays closed — so a
/// transient failure (no fixture, a panic, an unusable probe) is measured again
/// by the next refresh instead of closing the fast path for the life of the
/// process. Such a run is only ever attempted by a refresh, so its cost is
/// bounded by the reader's own cadence and by the battery's per-child deadline;
/// a shutdown starts no measurement and stops one in flight at its next row
/// boundary.
///
/// Returns whether a verdict for the environment in effect stands afterwards.
///
/// `env` must be the pairs a command gets right now — the same list
/// [`invalidate_verdict_for`] was given for the environment being adopted, or
/// [`super::super::agent_env_pairs`] when no read replaced it. The caller derives
/// it once and passes the same list to both, so the verdict, the key it is filed
/// under and what a command actually runs under cannot drift apart.
pub(crate) fn refresh_if_environment_changed(env: &[(OsString, OsString)]) -> bool {
    // On Windows there is no real search to run, so there is nothing to measure:
    // this arm is the gate's own statement of the rule. The reader checks the
    // platform too, so a failed read there does not ask the pool for this `false`.
    if cfg!(windows) {
        return false;
    }
    // Nothing new starts during a shutdown, and a battery already running stops at
    // its next row boundary ([`run_battery`]), so the reader's join can be held
    // only by the child in flight — bounded by [`BATTERY_SPAWN_TIMEOUT`].
    if crate::shutdown::aborting() {
        return ESTABLISHED.load().is_some();
    }
    let key = EnvKey::new(env);
    if LAST_MEASURED.lock().unwrap_poison().as_ref() == Some(&key) {
        return true;
    }
    // Fail-closed for the window the battery takes. The reader has normally done
    // this already, in the step that put the environment in place; the store
    // covers any other way it moved.
    ESTABLISHED.store(None);
    record(key, measure(env))
}

/// Drop the verdict in effect when the environment commands are about to get is
/// not the one it was measured for.
///
/// The reader calls this with the environment a fresh read will publish, in the
/// step *before* it publishes: the verdict is gone before any command can run
/// under the new environment, so the two never mix (a command served in between
/// still sees the old environment, which is the one the verdict belongs to). No
/// blocking, and nothing is measured here — [`refresh_if_environment_changed`]
/// does that on the blocking pool.
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

/// Record a measurement's outcome for `key`, and report whether a verdict is in
/// effect afterwards.
///
/// A measured verdict is published together with the key it was measured for; a
/// run that produced none claims nothing, clearing both the key and the gate, so
/// the next refresh measures again rather than a closed gate standing for the
/// life of the process.
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

/// Install a verdict for the calling thread.
///
/// The in-crate lanes that exercise the served path (`parity_tests`, and every
/// serve pin whose rows would be demoted by a closed gate) are about the engine's
/// own equivalence proof against the locale those lanes pin, not about the runtime
/// gate; the gate has its own tests below. The verdict rides a per-thread override
/// (see [`TEST_VERDICT`]) so a lane does not race a sibling lane's own publish.
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

/// Publish an all-established verdict for the e2e harness.
///
/// The harness's subject is the engine's own equivalence proof against the
/// pinned locale — it drives the served path end to end and diffs it against the
/// real grep — while the runtime gate has its own tests; so the harness opens the
/// gate for the whole of its own process, which neither measures nor refreshes.
/// This is the one verdict published without a key (the rule [`record`] applies to
/// measured ones): there is no environment here for a key to name. Nothing here
/// weakens the harness's assertions.
#[cfg(feature = "grep-engine-e2e")]
pub(super) fn publish_all_for_harness() {
    publish(Established::all());
}

// ── The differential battery ──────────────────────────────────────────────

/// The battery's tallies, and the verdict they fold into.
#[derive(Default)]
struct Battery {
    /// The capabilities this run proved for the locale it ran under. `None` when
    /// nothing usable was measured (an unusable probe, so no row was ever compared)
    /// or when a row that had to be compared could not be (see `unmeasured`): the
    /// run then establishes nothing and the caller records nothing.
    verdict: Option<Established>,
    /// Rows attempted / compared / skipped / not measurable, as the INFO line
    /// reports them.
    rows: usize,
    compared: usize,
    skipped: usize,
    /// Rows that were meant to be compared but could not be: one side did not run
    /// (the real search is missing or unusable, or the deadline expired). Counted
    /// apart from skipped rows, because such a row is evidence of *nothing* and
    /// must not let a group be established from the rows that did run.
    unmeasured: usize,
    /// What the probe that decides whether a real search exists here found.
    real_search: Probe,
}

/// One capability's row tally. A capability is established only when at least
/// one row was actually compared and no row failed — a disagreement, or a
/// capability the engine never served, or one with no rows at all, leaves it
/// fail-closed. A row that could not be compared at all is not this group's to
/// fail: it is counted on the battery, where it stops a verdict being folded from a
/// partial run (see [`Battery::fold`]).
#[derive(Default)]
struct Group {
    compared: usize,
    failed: bool,
}

impl Battery {
    /// Fold a run's groups into its verdict: every capability from its own group,
    /// and nothing at all when a row that had to be compared could not be. Such a
    /// run is not a verdict for the environment — it claims nothing, so the next
    /// read measures again rather than one transient row failure keeping a
    /// capability on the real search for the life of the process.
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
    /// No parity question: the engine cannot serve this shape (the gate would
    /// demote it anyway), or the row does not apply to this host.
    Skipped,
    /// The row had to be compared and could not be: one side did not run. Claims
    /// nothing about the capability it belongs to — it is counted on the battery,
    /// which then folds no verdict at all.
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

    /// Whether this capability came out established.
    const fn established(&self) -> bool {
        self.compared > 0 && !self.failed
    }
}

/// Measure every capability under `env`, with nothing allowed to panic out of the
/// run: a fixture that cannot be built, a panic inside, a probe that could not be
/// run, or a row that had to be compared and could not be yields no verdict at all,
/// and only booleans and counts are ever logged.
///
/// `None` is "no verdict could be taken", as opposed to a verdict that established
/// nothing: nothing is claimed for such a run, so the caller records no key for it
/// and measures again later.
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
    tracing::info!(
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

/// Run every row and fold the verdict.
///
/// A host with no real search at all answers immediately: the fast path is then
/// the only search there is, so it stands for every capability. A probe that
/// could not be *run* is not that case — it establishes nothing, so the verdict
/// stays `None` and the caller's fail-closed default stands. A shutdown begun
/// mid-run stops the rows at the next boundary, with the same effect.
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

    // A shutdown stops the battery at the next row boundary: the reader's join
    // waits for the whole run, so this is what keeps teardown from waiting for it.
    // The rows not run are simply not claimed — no verdict is folded — so the gate
    // stays closed and the next read measures again.
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
    // The trailing filter row runs a child too, so the shutdown check covers it
    // like any row above.
    if crate::shutdown::aborting() {
        return battery;
    }
    glob.record(
        &mut battery,
        compare_grep_row(fixture, env, &filter_row(fixture)),
    );

    // A row that could not be compared is evidence of nothing, and a run that could
    // not compare every row it set out to is not a verdict for this environment (see
    // [`Battery::fold`]).
    battery.fold(&plain, &general, &glob);
    battery
}

// ── Rows ──────────────────────────────────────────────────────────────────

/// One differential row: the verb-first argv both sides receive, plus whether
/// the row's stdout compares as a sorted record set (recursive walks only — the
/// parallel walk orders files across workers, an approved delta).
struct Row {
    argv: Vec<String>,
    sorted: bool,
}

impl Row {
    /// A row over `argv` (verb first), with absolute operand paths so both sides
    /// print the same `file:` prefixes.
    fn new(argv: &[&str], sorted: bool) -> Self {
        Row {
            argv: argv.iter().map(|word| (*word).to_string()).collect(),
            sorted,
        }
    }
}

/// The literal shapes: byte matching over the ascii, unicode, invalid-UTF-8 and
/// binary fixtures, every flag that does not put case or word boundaries between
/// the pattern and the bytes, one two-file row, and one clean recursive walk.
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

/// The shapes whose matching a locale reaches into: Unicode case folding over
/// the tricky code points, POSIX classes against non-ASCII data, `.` over the
/// multibyte data, a bracket set holding an accented character, a word boundary,
/// and `-i` over the invalid-UTF-8 and binary fixtures.
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

/// The glob operands the glob rows expand. The names tree holds an
/// ASCII-uppercase name, an accented pair and a plain name, so a locale's
/// collation and character classes have something to disagree about.
const GLOB_PATTERNS: &[&str] = &["*.txt", "?.txt", "[à]*.txt", "[a-z]*.txt", "[!a]*.txt"];

/// The filter row: `--include`'s `fnmatch` is read by the engine from a process
/// that never calls `setlocale` and by the real search under the locale in effect.
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

// ── Row comparison ────────────────────────────────────────────────────────

/// Compare one grep row end to end: same argv, same environment, engine
/// in-process against the real search.
///
/// A shape the engine cannot serve is no parity question ([`RowOutcome::Skipped`]
/// — the gate would demote it anyway), while a real search that did not run says
/// nothing about the row ([`RowOutcome::Unmeasured`]), so the capability stays
/// fail-closed instead of being established from the rows that did run.
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

/// Build the spec the production parent path builds for `segment`, so the
/// comparison exercises the served shape rather than a hand-built one. `None`
/// when that path refuses the member.
///
/// One deliberate deviation from a unix serve decision: `allow_single` is `true`
/// (the Windows production value), which bypasses the unix single-file perf gate.
/// A single-file lookup is exactly one of the shapes the engine must be held to
/// even though the unix path hands it to the real search for speed.
///
/// The call suspends the gate for its duration (see [`Measuring`]): the row under
/// measurement must be built whatever the verdict currently says.
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
            // The battery builds one member at a time, never a line that has
            // already exported something.
            env_changed: false,
        },
    )
    .ok()
    .map(|(spec, _redirects)| spec)
}

/// Compare [`expand_glob`] with the shell's own expansion of the same operand:
/// same members, same order. The same distinction as [`compare_grep_row`]: an
/// operand the engine cannot expand is no parity question, a shell that did not
/// run is nothing to measure.
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
/// `sh -c 'printf "%s\n" <pattern>'` under the agent's environment, cwd = the
/// names tree. `None` when `sh` could not be run at all, which measures nothing.
fn shell_glob(
    pattern: &str,
    fixture: &Fixture,
    env: &[(OsString, OsString)],
) -> Option<Vec<String>> {
    let script = format!("printf \"%s\\n\" {pattern}");
    let mut cmd = agent_command("sh", env);
    cmd.arg("-c").arg(&script).current_dir(&fixture.names);
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

/// Run the real search by name with the agent's environment applied and the
/// fixture root as cwd.
fn real_search(argv: &[String], env: &[(OsString, OsString)], cwd: &Path) -> Child {
    let Some((verb, operands)) = argv.split_first() else {
        // No program name at all: nothing ran, so nothing was measured.
        return Child::Unusable;
    };
    let mut cmd = agent_command(verb, env);
    cmd.args(operands).current_dir(cwd);
    bounded_output(&mut cmd)
}

/// A command with the agent's environment applied whole (nothing inherited), and
/// its stdin and stderr taken away: the battery runs the owner's own programs
/// unattended in the background, so nothing here may read the daemon's stdin or
/// spill its diagnostics into the daemon's stderr. Its stdout is the row's
/// capture ([`bounded_output`]).
///
/// The window guarantee `tools::shell::tree` documents holds for this site too
/// (the battery never runs on Windows, but the flag is what every spawn in the
/// tree carries); `creation_flags` is a Windows-only API, so — like the engine's
/// own probe — it rides its own `cfg`.
fn agent_command(program: &str, env: &[(OsString, OsString)]) -> Command {
    let mut cmd = Command::new(program);
    crate::tools::shell::apply_env_pairs(&mut cmd, env);
    cmd.stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    }
    cmd
}

/// One battery child's result: its stdout and exit code, or why it produced
/// neither. The distinction between the two failure arms is what
/// [`real_search_exists`] needs — "not installed here" and "could not be
/// started" are not the same finding.
enum Child {
    /// Ran to completion inside the deadline.
    Ran(Vec<u8>, i32),
    /// Could not be started because nothing in this environment's search path
    /// resolves to it: the program is not on the system.
    Missing,
    /// No usable result: it could not be started for any reason other than not
    /// being there, it failed to be waited on, or it outlived
    /// [`BATTERY_SPAWN_TIMEOUT`] and was killed.
    Unusable,
}

/// Run one battery child to completion under a deadline, and hand back what it
/// wrote.
///
/// A deadline is what keeps the battery — and so the environment reader that
/// hands the measurement to a blocking pool and waits for it — from depending on
/// a program the owner's own `PATH` resolves: a `grep` that hangs would otherwise
/// wedge a blocking thread and stop every later read. The child is killed on
/// expiry.
///
/// The child's stdout is a file, not a pipe, and that is what makes the deadline
/// cover the whole row: a pipe has to be read to EOF, and an EOF a stray holder of
/// the write end never produces would block this thread for good, while a file
/// needs no reader at all — once the child has exited, what it wrote is there.
/// A killed child is handed to [`reap`], so not even one that survives SIGKILL can
/// hold this thread. The capture is then bounded whatever the owner's program does:
/// what is read back is capped at [`BATTERY_STDOUT_CAP`], and a child that has
/// already written past it is killed rather than waited for. (Its stdout is still
/// not a terminal, so a program that formats for one is not what either side of a
/// comparison sees.)
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
        // "Not on this system" is a finding; any other spawn failure — a
        // permission refusal, a resource limit — is not evidence of absence.
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
            // Nothing a row needs comes anywhere near the cap, so a child already
            // past it is a runaway: killed, and the row is unusable rather than
            // captured.
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

/// Kill a battery child and collect it, under the same bound as the row it belongs
/// to.
///
/// The wait is bounded for the same reason the row's is: a child that survives
/// SIGKILL (uninterruptible I/O is the case the deadline exists for) must not park
/// the blocking thread — the environment reader joins that thread, so a parked one
/// would stop every later read. A child still running when the bound expires is left
/// to the OS rather than waited for.
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

/// Whether a real search exists here at all. Three-way, because "there is
/// nothing on this system to compare against" is a verdict of its own — the fast
/// path is then the only search and stands — while a probe that could not be
/// *run* says nothing about whether a real search is there, and must leave the
/// verdict fail-closed rather than publish one nothing was measured under.
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
    /// Nothing in this environment resolves to a real search: the fast path is
    /// the only search here, so it stands for every capability.
    Absent,
    /// The probe itself could not be run. Not evidence of absence: nothing is
    /// established, and the real search answers if one turns up.
    #[default]
    Unknown,
}

/// Run one spec in-process the way the existing in-process lanes do: the
/// production matcher builder, a buffer sink standing in for the capture pipe, and
/// empty stdin. `None` when the matcher cannot build — the parent path has
/// already validated the spec, so that is a fail-closed arm, not a parity row.
fn engine_run(spec: &EngineSpec) -> Option<(Vec<u8>, i32)> {
    let matcher = build_matcher(&spec.patterns, spec.mode, &spec.flags).ok()?;
    let mut out = Output::new(OutputSink::Buffer(Vec::new()), StdoutDest::of(spec, true));
    let consumed = std::cell::Cell::new(false);
    let code = serve_into(spec, &matcher, &mut out, io::empty(), &consumed);
    let (bytes, _err) = out.take_stdio();
    Some((bytes, code))
}

/// Byte-exact stdout plus an equal exit code, relaxed to sorted record sets for
/// the recursive rows (the parallel walk's cross-file order is an approved delta).
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

/// The shell segment the engine's parent-side reader is handed for a row: the
/// same argv words, shell-quoted, so the quoted spellings the tokenizer sees are
/// the words the real search receives.
fn segment_text(argv: &[String]) -> String {
    argv.iter()
        .map(|word| shell_quote(word))
        .collect::<Vec<_>>()
        .join(" ")
}

// ── Fixture ───────────────────────────────────────────────────────────────

/// The byte-exact fixture the battery searches: a scratch tree under the shell
/// temp root, removed when the run ends.
struct Fixture {
    /// Owns the tree: dropping the battery removes every fixture file.
    _tmp: tempfile::TempDir,
    /// The tree's root, which is also each grep row's cwd.
    root: PathBuf,
    /// A home directory, for the operand resolution the parent path runs.
    home: PathBuf,
    /// The tree the glob rows expand within (`names/`).
    names: PathBuf,
}

impl Fixture {
    /// Build the fixture tree inside [`crate::temp::shell_tmpdir`]. `None` when
    /// the directory or any fixture file cannot be created — the caller stays
    /// fail-closed.
    fn create() -> Option<Self> {
        let tmp = tempfile::tempdir_in(crate::temp::shell_tmpdir()).ok()?;
        let root = tmp.path().to_path_buf();
        std::fs::create_dir_all(root.join("dir/sub")).ok()?;
        std::fs::create_dir_all(root.join("names")).ok()?;
        std::fs::create_dir_all(root.join("home")).ok()?;
        // Byte-exact: the rows depend on these code points and on these invalid
        // bytes, not on any encoding a writer might apply. The `dir/` tree has no
        // hidden entries, no ignore files and no binary files, so the engine's
        // rg-default walk exclusions can never be the cause of a difference.
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

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::{Fallback, PipelineCtx, grep_tokenize, parse_grep_words, serve_one_grep};
    use super::*;

    /// Parse a grep segment exactly as the serve path does, so a `needs` row
    /// reads the same [`ParsedGrep`] the gate sees.
    fn parsed(segment: &str) -> ParsedGrep {
        let mut words = grep_tokenize(segment, ShellPlatform::Unix).expect("tokenize");
        words.remove(0);
        parse_grep_words(&words, "grep", ShellPlatform::Unix).expect("parse")
    }

    /// Each row outcome lands in its own tally, and a run that could not compare a
    /// row it had to folds no verdict at all — so the next read measures again
    /// instead of one transient row failure keeping a capability on the real search
    /// for the life of the process.
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

    /// The verdict and the key it was measured for move together: a verdict is in
    /// effect only while the environment it belongs to is, and a measurement that
    /// produced none claims nothing at all — so a transient failure cannot leave
    /// the gate closed for the life of the process, and an environment that moves
    /// loses the old verdict before it can be served under it.
    ///
    /// Driven directly, because a real measurement spawns the owner's own programs;
    /// the battery's own rows are measured by `battery_reports_capabilities_per_locale`.
    ///
    /// `#[serial]`: the verdict and its key are process-global.
    #[cfg(unix)]
    #[serial_test::serial(shell_env)]
    #[test]
    fn a_verdict_stands_only_for_the_environment_it_was_measured_for() {
        let in_effect = || EnvKey::new(&crate::tools::shell::agent_env_pairs());
        let other = [(OsString::from("ONE"), OsString::from("two"))];
        // The gate itself, not `established()`: the latter prefers the test-only
        // per-thread override, which another lane may have installed on this
        // thread, and this lane is about the process-global verdict.
        let gate = || ESTABLISHED.load().as_deref().copied().unwrap_or_default();

        // A measured environment is in effect, and looking again at the *same*
        // environment takes no measurement at all.
        assert!(record(in_effect(), Some(Established::all())));
        assert_eq!(gate(), Established::all());
        assert!(
            refresh_if_environment_changed(&crate::tools::shell::agent_env_pairs()),
            "an environment measured with a verdict is a no-op"
        );

        // A run that produced no verdict claims nothing: the gate closes and no key
        // is left, which is what makes the next refresh measure again.
        assert!(!record(in_effect(), None));
        assert_eq!(gate(), Established::default());
        assert!(LAST_MEASURED.lock().unwrap_poison().is_none());

        // An environment that is not the measured one drops the verdict in the
        // caller's own step — and leaves no key behind either.
        assert!(record(EnvKey::new(&other), Some(Established::all())));
        assert_eq!(gate(), Established::all());
        invalidate_verdict_for(&crate::tools::shell::agent_env_pairs());
        assert_eq!(gate(), Established::default());
        assert!(LAST_MEASURED.lock().unwrap_poison().is_none());

        // An environment that *is* the measured one keeps its verdict: the check is
        // about the environment, not about being called.
        assert!(record(EnvKey::new(&other), Some(Established::all())));
        invalidate_verdict_for(&other);
        assert_eq!(gate(), Established::all());
        assert!(LAST_MEASURED.lock().unwrap_poison().as_ref() == Some(&EnvKey::new(&other)));

        // Leave the process as this lane found it.
        ESTABLISHED.store(None);
        *LAST_MEASURED.lock().unwrap_poison() = None;
    }

    /// `needs` in one table: byte literals keep the fast path, anything whose
    /// matching a locale reaches into is general, and a glob operand or a filter
    /// adds the glob dimension.
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

    /// The gate demotes every capability that was not established, naming the
    /// unmet one; with everything established it serves.
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

    /// A host with no real search to compare against: the battery answers with
    /// every capability established — the fast path is the only search there —
    /// rather than with the fail-closed default a probe that could not be *run*
    /// leaves in place.
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

    /// The gate never fires on Windows: there is no real search to run there, so
    /// the fast path is the only search and stands however the verdict stands.
    /// The platform is a value, so this is drivable from this host.
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

    /// The battery's per-locale evidence: which capabilities come out established
    /// under the locale the commands would actually get, under `C`, and under a
    /// UTF-8 collating locale. Printing only — a host with no real search to
    /// compare against is reported and skipped rather than failed — plus the
    /// sanity check that the host's real search was there and rows ran.
    ///
    /// Ignored, like the repo's other real-search evidence lanes: it spawns the
    /// host's own grep and sh, which a build host need not have. Run it with
    ///
    /// ```text
    /// cargo test --lib -- --ignored --nocapture battery_reports_capabilities_per_locale
    /// ```
    #[cfg(unix)]
    #[test]
    #[ignore = "manual evidence run: prints the per-locale verdicts from the host's real search"]
    fn battery_reports_capabilities_per_locale() {
        let locales = [
            ("inherited", crate::tools::shell::agent_env_pairs()),
            ("C", locale_env("C")),
            ("en_US.UTF-8", locale_env("en_US.UTF-8")),
        ];
        for (label, env) in locales {
            let fixture = Fixture::create().expect("fixture");
            let battery = run_battery(&fixture, &env);
            // A run that established nothing prints as all-false: that is what the
            // gate serves in that case, and the probe's finding — or the unmeasured
            // tally, which folds no verdict at all — says why.
            let verdict = battery.verdict.unwrap_or_default();
            println!(
                "grep parity [{label}]: established={} plain={} general={} glob={} rows={} compared={} skipped={} unmeasured={} real_search={:?}",
                battery.verdict.is_some(),
                verdict.plain,
                verdict.general,
                verdict.glob,
                battery.rows,
                battery.compared,
                battery.skipped,
                battery.unmeasured,
                battery.real_search,
            );
            match battery.real_search {
                Probe::Present => {}
                Probe::Absent => {
                    println!(
                        "  no real search here to compare against: the fast path is the only search"
                    );
                    continue;
                }
                Probe::Unknown => {
                    println!(
                        "  the probe could not be run: nothing was established and the fast path is closed"
                    );
                    continue;
                }
            }
            // Which way each shape went — and, for every capability named as
            // served, that every compared row of it matched the real search.
            let (served, real): (Vec<Need>, Vec<Need>) = [Need::Plain, Need::General, Need::Glob]
                .into_iter()
                .partition(|need| verdict.covers(*need));
            println!(
                "  served by the fast path: {} | run by the real search: {}",
                capability_names(&served),
                capability_names(&real),
            );
            assert!(battery.rows > 0, "{label}: the battery ran no row");
        }
    }

    /// The capability names in a list, or `none`.
    #[cfg(unix)]
    fn capability_names(needs: &[Need]) -> String {
        if needs.is_empty() {
            return "none".to_string();
        }
        needs
            .iter()
            .map(|need| need.capability())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// An environment pair list for `locale`, with the inherited `PATH` kept so
    /// the real search is found.
    #[cfg(unix)]
    fn locale_env(locale: &str) -> Vec<(OsString, OsString)> {
        let path = std::env::var_os("PATH").unwrap_or_default();
        vec![
            (OsString::from("LC_ALL"), OsString::from(locale)),
            (OsString::from("PATH"), path),
        ]
    }
}
