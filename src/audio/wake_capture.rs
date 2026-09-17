//! Bench-only capture of the raw encoder token matrices the scoring pipeline
//! already computes and throws away, the product's own per-window statistic,
//! a per-clip detection ledger and the frozen A/B partition.
//!
//! Off by default; opt in with `MAHBOT_WAKE_CAPTURE=baseline|measure`.  The
//! output directory is `<storage root>/wake_capture`; `baseline` writes only
//! the non-token artifacts (per-clip ledger, partition, meta) so a baseline
//! pass reproduces the shipped numbers, while `measure` also writes the token
//! matrices.  The offline analyser that consumes these files is a separate
//! target and does not live here.
//!
//! The harness is single-threaded for the corpus phases, so the arming scope
//! and the lazily-opened output files are thread-local.  Writes go straight to
//! the `File` (no `BufWriter`) so a process kill cannot lose captured records;
//! a failed write marks the capture broken and stops it for the rest of the
//! process, and the reader tolerates a partial trailing record while a partial
//! record anywhere else is an error.

use rand::rngs::StdRng;
use rand::seq::SliceRandom as _;
use rand::{RngExt as _, SeedableRng as _};
use serde_json::{Map, Value, json};
use std::cell::RefCell;
use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::OnceLock;

// ── Mode ─────────────────────────────────────────────────────────────────

const MODE_ENV: &str = "MAHBOT_WAKE_CAPTURE";
/// Optional measurement-session id (`MAHBOT_WAKE_RUN`).
///
/// Launch both passes with the same value and the analyser can prove their
/// artefacts come from one session; unset leaves that linkage unasserted.
const RUN_ENV: &str = "MAHBOT_WAKE_RUN";

/// The family whose clips form the recall set.
pub(crate) const POSITIVE_FAMILY: &str = "positive";

/// The two passes of one measurement session, in run order.
pub(crate) const PASS_BASELINE: &str = "baseline";
pub(crate) const PASS_MEASURE: &str = "measure";
pub(crate) const PASSES: [&str; 2] = [PASS_BASELINE, PASS_MEASURE];

/// Capture artefact names — the analyser reads them through these, never
/// through a repeated literal.
pub(crate) const FILE_PARTITION: &str = "partition.jsonl";
pub(crate) const FILE_WINDOWS: &str = "windows.jsonl";
pub(crate) const FILE_ENROLLMENT: &str = "enrollment.jsonl";
pub(crate) const FILE_TOKENS: &str = "tokens.bin";

/// `clip_results_<pass>.jsonl` — one pass's per-clip detection ledger.
pub(crate) const FILE_CLIP_RESULTS_BASELINE: &str = "clip_results_baseline.jsonl";
pub(crate) const FILE_CLIP_RESULTS_MEASURE: &str = "clip_results_measure.jsonl";

/// The per-clip ledger of one pass.  A caller passes one of [`PASSES`].
///
/// # Panics
/// Panics on any other `pass` — nothing builds these names from a free-form
/// string, so a mismatch is a bug rather than a capture condition.
#[must_use]
pub(crate) fn clip_results_file(pass: &str) -> &'static str {
    match pass {
        PASS_BASELINE => FILE_CLIP_RESULTS_BASELINE,
        PASS_MEASURE => FILE_CLIP_RESULTS_MEASURE,
        other => panic!("{other} is not one of the two measurement passes"),
    }
}

/// `meta_<pass>.json` — one pass's manifest.
#[must_use]
pub(crate) fn meta_file(pass: &str) -> String {
    format!("meta_{pass}.json")
}

/// The measurement-session id from `MAHBOT_WAKE_RUN`, read once per process.
#[must_use]
pub(crate) fn run_id() -> Option<&'static str> {
    static RUN: OnceLock<Option<String>> = OnceLock::new();
    RUN.get_or_init(|| std::env::var(RUN_ENV).ok().filter(|id| !id.is_empty()))
        .as_deref()
}

/// Fixed at partition time — the analyser treats this half as the selection
/// set.
pub(crate) const SELECTION_HALF: &str = "A";

/// Seed for the only nondeterminism in the partition rule: the label-ordering
/// tie break and the level-half tie break.
const TIE_SEED: u64 = 20_260_917;

/// Capture mode selected by `MAHBOT_WAKE_CAPTURE`.  `None` (the default) is
/// capture off.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mode {
    Baseline,
    Measure,
}

impl Mode {
    /// The `pass` label and filename suffix of this pass.
    #[must_use]
    pub(crate) const fn pass_name(self) -> &'static str {
        match self {
            Mode::Baseline => PASS_BASELINE,
            Mode::Measure => PASS_MEASURE,
        }
    }
}

/// `MAHBOT_WAKE_CAPTURE`, read once per process (ASCII case-insensitive).
///
/// `None` means capture off — either because the variable is unset or `off`.
/// Unrecognised and non-UTF-8 values also fall back to `None` with a warning
/// naming the value.
#[must_use]
pub(crate) fn mode() -> Option<Mode> {
    static MODE: OnceLock<Option<Mode>> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var(MODE_ENV) {
        Ok(raw) => match raw.to_ascii_lowercase().as_str() {
            "baseline" => Some(Mode::Baseline),
            "measure" => Some(Mode::Measure),
            "off" => None,
            _ => {
                eprintln!(
                    "{MODE_ENV}={raw:?}: unrecognised capture mode — capture off \
                     (expected off|{})",
                    PASSES.join("|")
                );
                None
            }
        },
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(raw)) => {
            eprintln!(
                "{MODE_ENV}={}: non-UTF-8 capture mode — capture off",
                raw.display()
            );
            None
        }
    })
}

/// Whether the capture records token data (`MAHBOT_WAKE_CAPTURE=measure`).
#[must_use]
pub(crate) fn measuring() -> bool {
    mode() == Some(Mode::Measure)
}

/// Whether any capture artifact is written (`baseline` or `measure`).
#[must_use]
pub(crate) fn active() -> bool {
    mode().is_some()
}

/// The current pass's label, or `None` when capture is off.
#[must_use]
pub(crate) fn pass_name() -> Option<&'static str> {
    mode().map(Mode::pass_name)
}

/// Output directory: `<storage root>/wake_capture`.
///
/// # Panics
/// Panics when the storage root is not set — the bench sets it before any
/// capture call.
#[must_use]
pub(crate) fn capture_dir() -> PathBuf {
    crate::config::CONFIG
        .try_storage_root()
        .expect("wake capture requires the storage root to be set")
        .join("wake_capture")
}

// ── Arming scope ─────────────────────────────────────────────────────────

/// What the current thread is capturing.  Windows scored with no scope
/// (enrollment-consistency self-test, negative calibration, warm-up audio) are
/// never captured — the scope is the only exclusion mechanism.
#[derive(Default)]
enum Scope {
    #[default]
    None,
    Enrollment,
    Clip {
        family: Rc<str>,
        label: Rc<str>,
    },
}

impl Scope {
    /// `(family, label)` when the scope is a clip.
    fn clip(&self) -> Option<(&Rc<str>, &Rc<str>)> {
        match self {
            Scope::Clip { family, label } => Some((family, label)),
            Scope::None | Scope::Enrollment => None,
        }
    }
}

/// A token matrix stashed by [`on_window_tokens`], waiting for its paired
/// [`on_window_scored`].
struct Pending {
    family: Rc<str>,
    label: Rc<str>,
    tokens: Vec<f32>,
    token_count: usize,
    dim: usize,
}

// ── Output files ─────────────────────────────────────────────────────────

/// The capture files, each lazily opened (the first open truncates it).
#[derive(Default)]
struct Files {
    partition: Option<File>,
    windows: Option<File>,
    tokens: Option<File>,
    enrollment: Option<File>,
    clip_results: Option<File>,
}

/// One capture file.  `ClipResults` carries the current pass's label because
/// its filename is per pass; the rest are fixed names.
#[derive(Clone, Copy)]
enum Target {
    Partition,
    Windows,
    Tokens,
    Enrollment,
    ClipResults(&'static str),
}

impl Target {
    fn name(self) -> &'static str {
        match self {
            Target::Partition => FILE_PARTITION,
            Target::Windows => FILE_WINDOWS,
            Target::Tokens => FILE_TOKENS,
            Target::Enrollment => FILE_ENROLLMENT,
            Target::ClipResults(pass) => clip_results_file(pass),
        }
    }

    fn slot(self, files: &mut Files) -> &mut Option<File> {
        match self {
            Target::Partition => &mut files.partition,
            Target::Windows => &mut files.windows,
            Target::Tokens => &mut files.tokens,
            Target::Enrollment => &mut files.enrollment,
            Target::ClipResults(_) => &mut files.clip_results,
        }
    }
}

#[derive(Default)]
struct State {
    scope: Scope,
    flush: bool,
    clip_pos: u64,
    pending: Option<Pending>,
    tokens_offset: u64,
    windows_captured: u64,
    flush_windows: u64,
    enrollment_utterances: u64,
    unpaired_windows: u64,
    broken: bool,
    files: Files,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

/// The capture output directory, created once per process.  A creation failure
/// is reported once; the file writes that follow report their own failures.
#[must_use]
fn ensure_capture_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = capture_dir();
        if let Err(e) = std::fs::create_dir_all(&dir) {
            eprintln!("wake capture: cannot create {}: {e}", dir.display());
        }
        dir
    })
}

impl State {
    /// Append raw bytes to one capture file, opening it on first use.  Any
    /// failure is reported and swallowed (`false`): the caller marks the
    /// capture broken, which ends it for the rest of the process.
    fn write_to(&mut self, target: Target, bytes: &[u8]) -> bool {
        let name = target.name();
        let path = ensure_capture_dir().join(name);
        let slot = target.slot(&mut self.files);
        if slot.is_none() {
            match File::create(&path) {
                Ok(file) => *slot = Some(file),
                Err(e) => {
                    eprintln!("wake capture: cannot create {name}: {e}");
                    return false;
                }
            }
        }
        match slot.as_mut().expect("just opened").write_all(bytes) {
            Ok(()) => true,
            Err(e) => {
                eprintln!("wake capture: write to {name} failed: {e}");
                false
            }
        }
    }

    /// Append one JSON line.  A failed write ends the capture for the rest of
    /// the process (the records written before it stay usable).
    fn append_record(&mut self, target: Target, value: &Value) {
        if self.broken {
            return;
        }
        let mut line = serde_json::to_string(value).expect("capture record is serializable");
        line.push('\n');
        if !self.write_to(target, line.as_bytes()) {
            self.broken = true;
        }
    }

    /// Append one token matrix as little-endian `f32`s and return its byte
    /// offset into `tokens.bin`, or `None` when the write failed — the capture
    /// is marked broken, so no later record can reference a misaligned offset.
    fn append_tokens(&mut self, tokens: &[f32]) -> Option<u64> {
        if self.broken {
            return None;
        }
        let mut bytes = Vec::with_capacity(tokens.len() * 4);
        for value in tokens {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let offset = self.tokens_offset;
        if !self.write_to(Target::Tokens, &bytes) {
            self.broken = true;
            return None;
        }
        self.tokens_offset += bytes.len() as u64;
        Some(offset)
    }
}

// ── Arming ───────────────────────────────────────────────────────────────

/// Capture the enrolment utterances encoded while the scope is armed.
pub(crate) fn arm_enrollment() {
    if !active() {
        return;
    }
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.scope = Scope::Enrollment;
        state.pending = None;
    });
}

/// Capture every window scored for one clip (`family`/`label` land in the
/// ledger).
pub(crate) fn arm_clip(family: &str, label: &str) {
    if !active() {
        return;
    }
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.scope = Scope::Clip {
            family: family.into(),
            label: label.into(),
        };
        state.clip_pos = 0;
        state.pending = None;
    });
}

/// Stop capturing (windows scored with no scope are never captured).
pub(crate) fn disarm() {
    if !active() {
        return;
    }
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.scope = Scope::None;
        state.pending = None;
    });
}

/// Mark whether subsequently captured windows are trailing-silence flush
/// windows; no-op unless the mode is `measure`.
pub(crate) fn mark_flush(flush: bool) {
    if !measuring() {
        return;
    }
    STATE.with(|state| {
        state.borrow_mut().flush = flush;
    });
}

// ── Capture hooks (shipped scoring path) ─────────────────────────────────

/// Number of encoder tokens to capture: the model's token count clamped to
/// what `tokens` actually holds.
fn captured_token_count(tokens: &[f32], token_count: usize, dim: usize) -> usize {
    if dim == 0 {
        return 0;
    }
    token_count.min(tokens.len() / dim)
}

/// Capture one encoder token matrix from
/// [`encode_window`](crate::audio::wake_word::encode_window).
///
/// Appends an enrolment record while the scope is [`Scope::Enrollment`], or
/// stashes the window for its paired [`on_window_scored`] while the scope is a
/// clip.  No-op unless the mode is `measure`.
pub(crate) fn on_window_tokens(tokens: &[f32], token_count: usize, dim: usize) {
    if !measuring() {
        return;
    }
    let count = captured_token_count(tokens, token_count, dim);
    if count == 0 {
        return;
    }
    let matrix = &tokens[..count * dim];
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let enrollment = matches!(state.scope, Scope::Enrollment);
        let clip = state
            .scope
            .clip()
            .map(|(family, label)| (Rc::clone(family), Rc::clone(label)));
        if enrollment {
            if let Some(offset) = state.append_tokens(matrix) {
                let record = json!({
                    "utterance": state.enrollment_utterances,
                    "tokens": count,
                    "dim": dim,
                    "offset": offset,
                });
                state.enrollment_utterances += 1;
                state.append_record(Target::Enrollment, &record);
            }
        } else if let Some((family, label)) = clip {
            if state.pending.is_some() {
                // A window whose tokens were replaced before it was scored —
                // the capture's integrity report must show it.
                state.unpaired_windows += 1;
            }
            state.pending = Some(Pending {
                family,
                label,
                tokens: matrix.to_vec(),
                token_count: count,
                dim,
            });
        }
    });
}

/// Capture the product's own per-window statistic and close the window opened
/// by [`on_window_tokens`].
///
/// Writes the raw bytes to `tokens.bin` before the `windows.jsonl` ledger line
/// that references them, so a truncated tail can never point at missing data.
/// No-op unless the mode is `measure` and a window is pending — never panics.
pub(crate) fn on_window_scored(product_value: f32) {
    if !measuring() {
        return;
    }
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(pending) = state.pending.take() else {
            return;
        };
        let Some(offset) = state.append_tokens(&pending.tokens) else {
            return;
        };
        let pos = state.clip_pos;
        state.clip_pos += 1;
        state.windows_captured += 1;
        if state.flush {
            state.flush_windows += 1;
        }
        let record = json!({
            "clip": &*pending.label,
            "family": &*pending.family,
            "pos": pos,
            "flush": state.flush,
            "tokens": pending.token_count,
            "dim": pending.dim,
            "offset": offset,
            "product_value": product_value,
        });
        state.append_record(Target::Windows, &record);
    });
}

// ── Per-clip ledger ──────────────────────────────────────────────────────

/// Append one per-clip detection result to `clip_results_<pass>.jsonl` (works
/// in `baseline` mode too — that is how the baseline pass's per-clip results
/// are preserved).
pub(crate) fn record_clip_result(family: &str, label: &str, detected: bool) {
    let Some(mode) = mode() else {
        return;
    };
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let record = json!({
            "family": family,
            "label": label,
            "detected": detected,
            "run": run_id(),
        });
        state.append_record(Target::ClipResults(mode.pass_name()), &record);
    });
}

// ── Partition ────────────────────────────────────────────────────────────

/// One clip's frozen partition assignment.
pub(crate) struct Assigned {
    label: String,
    key: String,
    half: &'static str,
}

/// Phrase key used to group renditions of one phrase.
fn phrase_key(family: &str, label: &str) -> String {
    match family {
        POSITIVE_FAMILY => "phrase".to_string(),
        "confusable" | "unrelated" => {
            for prefix in ["confusable2_", "confusable_", "unrelated2_", "unrelated_"] {
                if let Some(rest) = label.strip_prefix(prefix) {
                    return rest.to_string();
                }
            }
            label.to_string()
        }
        _ => label.to_string(),
    }
}

/// Partition one family's clips into halves `"A"`/`"B"`.
///
/// The frozen rule (the analyser consumes [`write_partition`]'s output):
/// 1. Every clip has a stable identifier (its label) and a phrase key:
///    `positive` → the constant `"phrase"`; `confusable`/`unrelated` → the
///    label minus its `confusable_`/`confusable2_`/`unrelated_`/`unrelated2_`
///    band prefix (so the two band renditions of one phrase share a key);
///    `silence`/`noise` → the label.
/// 2. Order clips by label byte order.  Ordering ties cannot occur with
///    today's labels; a deterministic shuffle seeded by [`TIE_SEED`] keeps the
///    rule total if they ever do.
/// 3. Groups of ≥2 renditions of one phrase are assigned, in that order,
///    round-robin `A`, `B`, `A`, … so every phrase appears in both halves.
/// 4. The remaining single-rendition clips, in that order, go to the half that
///    currently holds fewer clips; a level tie is resolved by one draw from the
///    [`TIE_SEED`] RNG.
#[must_use]
pub(crate) fn partition_family(family: &str, labels: &[String]) -> Vec<Assigned> {
    let mut rng = StdRng::seed_from_u64(TIE_SEED);

    // 2. Stable order by label bytes, with the deterministic tie shuffle.
    let mut order: Vec<usize> = (0..labels.len()).collect();
    order.sort_by(|&a, &b| labels[a].as_bytes().cmp(labels[b].as_bytes()));
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && labels[order[j]] == labels[order[i]] {
            j += 1;
        }
        if j - i > 1 {
            order[i..j].shuffle(&mut rng);
        }
        i = j;
    }

    // 3. Group renditions of one phrase (the two bands are not adjacent in
    // label order), preserving encounter order inside each group.
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for &index in &order {
        let key = phrase_key(family, &labels[index]);
        if let Some((_, members)) = groups.iter_mut().find(|(k, _)| *k == key) {
            members.push(index);
        } else {
            groups.push((key, vec![index]));
        }
    }

    let mut halves: Vec<Option<&'static str>> = vec![None; labels.len()];
    let mut count_a = 0usize;
    let mut count_b = 0usize;
    for (_, members) in &groups {
        if members.len() < 2 {
            continue;
        }
        for (k, &index) in members.iter().enumerate() {
            let half = if k % 2 == 0 { "A" } else { "B" };
            halves[index] = Some(half);
            if half == "A" {
                count_a += 1;
            } else {
                count_b += 1;
            }
        }
    }

    // 4. Single-rendition clips, in stable order.
    for &index in &order {
        if halves[index].is_some() {
            continue;
        }
        let half = if count_a < count_b {
            "A"
        } else if count_b < count_a {
            "B"
        } else if rng.random_bool(0.5) {
            "A"
        } else {
            "B"
        };
        halves[index] = Some(half);
        if half == "A" {
            count_a += 1;
        } else {
            count_b += 1;
        }
    }

    order
        .iter()
        .map(|&index| Assigned {
            label: labels[index].clone(),
            key: phrase_key(family, &labels[index]),
            half: halves[index].expect("every clip is assigned to one of the two halves"),
        })
        .collect()
}

/// Append one line per clip to `<capture_dir>/partition.jsonl`.  The first
/// call in a process truncates the file; later calls append.
pub(crate) fn write_partition(family: &str, assigned: &[Assigned]) {
    if !active() {
        return;
    }
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        for entry in assigned {
            let record = json!({
                "family": family,
                "key": entry.key,
                "label": entry.label,
                "half": entry.half,
            });
            state.append_record(Target::Partition, &record);
        }
    });
}

// ── Reporting ────────────────────────────────────────────────────────────

/// Windows captured (measure mode only).
#[must_use]
pub(crate) fn windows_captured() -> u64 {
    STATE.with(|state| state.borrow().windows_captured)
}

/// Trailing-silence flush windows captured (measure mode only).
#[must_use]
pub(crate) fn flush_windows() -> u64 {
    STATE.with(|state| state.borrow().flush_windows)
}

/// Enrolment utterances captured (measure mode only).
#[must_use]
pub(crate) fn enrollment_utterances() -> u64 {
    STATE.with(|state| state.borrow().enrollment_utterances)
}

/// Windows whose token matrix was replaced before it was scored.
#[must_use]
pub(crate) fn unpaired_windows() -> u64 {
    STATE.with(|state| state.borrow().unpaired_windows)
}

/// Whether any capture write failed (which ends capture for the rest of the
/// process; the records written before it stay usable).
#[must_use]
pub(crate) fn broken() -> bool {
    STATE.with(|state| state.borrow().broken)
}

/// Write `<capture_dir>/meta_<pass>.json` (truncating any previous content).
///
/// Called once per pass at the end of the corpus phases; `corpus` carries the
/// actual per-family counts so a short corpus or any skipped clip is visible in
/// the artefact.  No-op when capture is off.
pub(crate) fn write_meta(corpus: &[(&str, usize)]) {
    let Some(mode) = mode() else {
        return;
    };
    let dir = ensure_capture_dir();

    let mut corpus_map = Map::new();
    for (family, count) in corpus {
        corpus_map.insert((*family).to_string(), json!(count));
    }

    let value = json!({
        "pass": mode.pass_name(),
        "run": run_id(),
        "broken": broken(),
        "unpaired_windows": unpaired_windows(),
        "selection_half": SELECTION_HALF,
        "capture_dir": dir.display().to_string(),
        "corpus": corpus_map,
        "enrollment_utterances": enrollment_utterances(),
        "windows_captured": windows_captured(),
        "flush_windows": flush_windows(),
        "timestamp": crate::db::now(),
    });
    let text = serde_json::to_string_pretty(&value).expect("meta is serializable");
    write_file(&dir.join(meta_file(mode.pass_name())), &text);
}

/// Write `text` to `path`, truncating it, reporting any failure (never
/// panicking — the capture reports, it does not abort the run).
fn write_file(path: &Path, text: &str) {
    match File::create(path) {
        Ok(mut file) => {
            if let Err(e) = file.write_all(text.as_bytes()) {
                eprintln!("wake capture: write to {} failed: {e}", path.display());
            }
        }
        Err(e) => eprintln!("wake capture: cannot create {}: {e}", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Every label lands in exactly one half, and the halves are `A`/`B`.
    fn assert_total(assigned: &[Assigned], labels: &[String]) {
        let got: BTreeSet<&str> = assigned.iter().map(|clip| clip.label.as_str()).collect();
        let want: BTreeSet<&str> = labels.iter().map(String::as_str).collect();
        assert_eq!(got, want, "every label is assigned exactly once");
        assert!(
            assigned
                .iter()
                .all(|clip| clip.half == "A" || clip.half == "B")
        );
    }

    fn half_of<'a>(assigned: &'a [Assigned], label: &str) -> &'a str {
        assigned
            .iter()
            .find(|clip| clip.label == label)
            .unwrap_or_else(|| panic!("{label} was partitioned"))
            .half
    }

    fn fingerprint(assigned: &[Assigned]) -> Vec<(String, String, String)> {
        assigned
            .iter()
            .map(|clip| (clip.label.clone(), clip.key.clone(), clip.half.to_string()))
            .collect()
    }

    /// The frozen A/B partition rule every reported number depends on: renditions
    /// of one phrase split across the halves, single-rendition clips level the
    /// two halves, and the result is total and deterministic.
    #[test]
    fn partition_family_rule() {
        // Positive renditions all share the `"phrase"` key, so they alternate
        // A/B — the halves hold the same count (give or take one).
        let positives: Vec<String> = (0..5).map(|index| format!("utt_{index:02}")).collect();
        let assigned = partition_family(POSITIVE_FAMILY, &positives);
        assert!(assigned.iter().all(|clip| clip.key == "phrase"));
        let count = |half: &str| assigned.iter().filter(|clip| clip.half == half).count();
        assert!(count("A") > 0 && count("B") > 0, "both halves hold clips");
        assert!(
            count("A").abs_diff(count("B")) <= 1,
            "renditions alternate A/B"
        );
        assert_total(&assigned, &positives);

        // The two band renditions of one phrase share a key, so they land in
        // DIFFERENT halves; the two single-rendition labels then even the halves
        // out, so they differ too.
        let confusable: Vec<String> = ["confusable_alpha", "confusable2_alpha", "beta", "zeta"]
            .map(str::to_string)
            .to_vec();
        let assigned = partition_family("confusable", &confusable);
        assert_ne!(
            half_of(&assigned, "confusable_alpha"),
            half_of(&assigned, "confusable2_alpha"),
        );
        assert_ne!(half_of(&assigned, "beta"), half_of(&assigned, "zeta"));
        assert_total(&assigned, &confusable);

        // Deterministic: the same labels always partition the same way (this
        // pins the seeded tie-break).
        assert_eq!(
            fingerprint(&assigned),
            fingerprint(&partition_family("confusable", &confusable)),
        );
    }
}
