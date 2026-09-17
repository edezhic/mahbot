//! Offline analyser for the bench-only wake-word measurement capture.
//!
//! [`crate::audio::wake_capture`] records, during a bench pass, the raw encoder
//! token matrices the scoring path already computes, the frozen A/B partition,
//! the per-clip detection ledgers and the pass metadata.  This module reads
//! those files back and turns the token matrices into every reported curve:
//! per-window readouts, per-clip frame reductions, the fixed decision-level
//! sweep, the fixed-recall working points, AUC, the pass-to-pass difference and
//! the candidate verdicts against the control.
//!
//! Reads only: no ASR/TTS models, no stores, no network.  The only write is
//! `analysis.json`, inside the capture directory.
//!
//! Nothing here is fitted from the data — the sweep bounds, the recall targets
//! and the candidate list are compile-time constants, and the partition is
//! consumed verbatim.
//!
//! The capture is read defensively but never repaired.  A partial trailing
//! record — the writer appends unbuffered, so a killed process can leave one —
//! is tolerated and reported, while a partial record anywhere else is a hard
//! error; a window whose bytes the token blob does not hold is skipped and
//! reported through `skipped_windows`.  Beyond that, the two passes' ledgers
//! and manifests must agree on one session id, each manifest must agree with
//! the files it names and with the convention it was written under, and the
//! kept records' byte ranges must account for `tokens.bin`: every disagreement
//! is reported, never corrected, except three.  A session-id disagreement is
//! fatal because it means the capture was mixed; a manifest that reports a
//! failed capture write (`broken`) or that names another pass or selection half
//! is fatal because that pass's ledger is silently short or describes a
//! different capture; and an enrolment-less capture is fatal because every
//! vector readout would score 0.  The manifest's unpaired-window count —
//! windows whose token matrix was replaced before it was scored — is reported
//! alongside those checks.
//!
//! Token units: `mean` and `magnitude_weighted_mean` reduce the window's raw
//! token rows (the weighted mean's weights are the raw token norms);
//! `component_max` and `last_k_*` reduce the window's L2-normalised tokens.
//! The resulting vector is normalised before the cosine either way.
//!
//! The comparison space is the product's own [`cosine_similarity`] — normalised
//! vectors clamped to `[0, 1]`.  The frozen sweep still spans −1.0 … 1.0, but
//! its `[-1, 0)` half fires nothing because no compared value can be negative:
//! no floor, threshold or veto is involved.
//!
//! Decision levels are built once, on the selection half, per candidate, then
//! applied unchanged to the held-out half: a plateau maps to its tightest
//! (highest) level, and an unattainable target falls back to the most permissive
//! level (−1.0) on the half that maps it.

use crate::audio::wake_capture::{
    FILE_ENROLLMENT, FILE_PARTITION, FILE_TOKENS, FILE_WINDOWS, PASS_BASELINE, PASS_MEASURE,
    PASSES, POSITIVE_FAMILY, SELECTION_HALF, capture_dir, clip_results_file, meta_file,
};
use crate::audio::wake_word::l2_normalize_in_place;
use crate::vector::cosine_similarity;
use anyhow::{Context as _, Result, anyhow};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write as _;
use std::ops::Range;
use std::path::{Path, PathBuf};

// ── Sweep geometry ───────────────────────────────────────────────────────

/// Swept decision levels: −1.0 … 1.0 in steps of 0.001.
const LEVEL_STEPS: usize = 2001;

/// Sweep index of level 0.0 (levels are `index/1000 − 1`).
const LEVEL_ZERO_INDEX: usize = 1000;

/// Levels per unit of sweep index.
const LEVEL_SCALE: f32 = 1000.0;

/// Fixed recall targets.  Built once on the selection half, then applied
/// unchanged to the held-out half.
const FIXED_RECALLS: [f32; 6] = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5];

/// Recall steps of the AUC grid (`0.00, 0.01, …, 1.00`).
const AUC_GRID_STEPS: usize = 100;

/// Recall steps of the compact reported curve (`0.05, 0.10, …, 1.00`).
const CURVE_GRID_STEPS: usize = 20;

/// Rule one's resolution: a movement of this many summed firing-negative counts
/// or fewer on the held-out half is taken as inside the resolution of this data.
///
/// The movement is measured in counts, and a clip may fire at up to six fixed
/// levels, so one differing clip is worth between one and six counts.  Reading
/// the ticket's "two clips or fewer" as two *counts* is the only band under
/// which rule one's literal claim holds for every movement it accepts: a
/// movement of two counts or fewer certainly comes from at most two clips.
const MOVEMENT_RESOLUTION: i64 = 2;

/// The control candidate — the product's own pooling under the streaming
/// reduction.  Every verdict is expressed relative to it.
const CONTROL: Candidate = Candidate {
    readout: Readout::Mean,
    reduction: Reduction::Sliding3,
};

// ── Readouts ─────────────────────────────────────────────────────────────

/// How one window's `tokens × dim` matrix is turned into one comparison value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Readout {
    /// Component-wise mean over the window's tokens.
    Mean,
    /// Component-wise maximum over the window's normalised tokens.
    ComponentMax,
    /// Component-wise mean over the window's last `k` normalised tokens
    /// (`k = 1` is the last token).
    LastK(u8),
    /// Component-wise mean weighted by each token's L2 norm, computed before
    /// any normalization.
    MagnitudeWeightedMean,
    /// Best cosine of each window token against the enrolment token pool,
    /// reduced by maximum.
    TokenBestMax,
    /// The same per-token bests, reduced by mean.
    TokenBestMean,
}

impl Readout {
    /// The eight reported readout curves, in the fixed enumeration order the
    /// selection tie-break uses.
    const ALL: [Self; 8] = [
        Self::Mean,
        Self::ComponentMax,
        Self::LastK(1),
        Self::LastK(2),
        Self::LastK(3),
        Self::MagnitudeWeightedMean,
        Self::TokenBestMax,
        Self::TokenBestMean,
    ];

    #[must_use]
    fn name(self) -> String {
        match self {
            Self::Mean => "mean".to_string(),
            Self::ComponentMax => "component_max".to_string(),
            Self::LastK(k) => format!("last_k_{k}"),
            Self::MagnitudeWeightedMean => "magnitude_weighted_mean".to_string(),
            Self::TokenBestMax => "token_best_max".to_string(),
            Self::TokenBestMean => "token_best_mean".to_string(),
        }
    }

    /// Scalar readouts have no window vector and no prototype: their value is
    /// computed directly from the window's tokens.
    #[must_use]
    const fn is_scalar(self) -> bool {
        matches!(self, Self::TokenBestMax | Self::TokenBestMean)
    }

    /// The readout entry a `last_k` curve rolls up to for the extra frame
    /// reductions: the `k = 1` curve.
    #[must_use]
    const fn representative(self) -> Self {
        match self {
            Self::LastK(_) => Self::LastK(1),
            other => other,
        }
    }

    /// Slot of this readout in [`Readout::ALL`] (the window-value layout).
    #[must_use]
    fn index(self) -> usize {
        Self::ALL
            .iter()
            .position(|entry| *entry == self)
            .expect("every computed readout is a Readout::ALL entry")
    }

    /// The number of reported readouts — the width of every window-value array.
    /// Derived from [`Readout::ALL`] so the enumeration order and the widths
    /// cannot drift.
    const COUNT: usize = Self::ALL.len();
}

// ── Frame reductions ─────────────────────────────────────────────────────

/// How one clip's non-flush per-window values are reduced to trigger values.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Reduction {
    /// Sliding mean of three consecutive windows.
    Sliding3,
    /// The clip's single maximum.
    ClipMax,
    /// The clip's single mean.
    ClipMean,
    /// The clip's single median.
    ClipMedian,
}

impl Reduction {
    #[must_use]
    const fn name(self) -> &'static str {
        match self {
            Self::Sliding3 => "sliding3",
            Self::ClipMax => "clip_max",
            Self::ClipMean => "clip_mean",
            Self::ClipMedian => "clip_median",
        }
    }
}

/// One `(readout, reduction)` candidate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Candidate {
    readout: Readout,
    reduction: Reduction,
}

/// Step one's candidate list: every readout under the streaming reduction.
///
/// Step 2 selects from exactly this list — the clip-level reductions do not
/// exist at that point — and step 3's [`step_three_candidates`] builds on it.
#[must_use]
fn step_one_candidates() -> Vec<Candidate> {
    Readout::ALL
        .iter()
        .map(|&readout| Candidate {
            readout,
            reduction: Reduction::Sliding3,
        })
        .collect()
}

/// Step three's extras: the three clip-level reductions for the reference
/// readout and for the step-2 winner, deduplicated — the two readouts coincide
/// when the winner is `mean`.
#[must_use]
fn step_three_candidates(best: Readout) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    for readout in [Readout::Mean, best.representative()] {
        for reduction in [
            Reduction::ClipMax,
            Reduction::ClipMean,
            Reduction::ClipMedian,
        ] {
            let candidate = Candidate { readout, reduction };
            if !candidates.contains(&candidate) {
                candidates.push(candidate);
            }
        }
    }
    candidates
}

/// Reduce one clip's per-window values to trigger values: the clip fires at
/// decision level `L` iff ANY trigger value is `>= L`.
#[must_use]
fn triggers(reduction: Reduction, values: &[f32]) -> Vec<f32> {
    match reduction {
        Reduction::Sliding3 => sliding3(values),
        Reduction::ClipMax => values
            .iter()
            .copied()
            .reduce(f32::max)
            .into_iter()
            .collect(),
        Reduction::ClipMean => mean_of(values).into_iter().collect(),
        Reduction::ClipMedian => median_of(values).into_iter().collect(),
    }
}

/// Sliding mean of three consecutive windows.  A clip with fewer than three
/// non-flush windows contributes the mean over the windows it has; a clip with
/// none contributes nothing, so it fires at no level.
#[must_use]
fn sliding3(values: &[f32]) -> Vec<f32> {
    if values.len() < 3 {
        return mean_of(values).into_iter().collect();
    }
    values.windows(3).filter_map(mean_of).collect()
}

// ── Numeric helpers ──────────────────────────────────────────────────────

/// `numerator / denominator` (0.0 for an empty denominator).
#[expect(
    clippy::cast_precision_loss,
    reason = "clip counts are tiny — exact in f32"
)]
#[must_use]
fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        return 0.0;
    }
    numerator as f32 / denominator as f32
}

/// `n` as `f32`; clip counts are far below f32's exact integer range.
#[expect(
    clippy::cast_precision_loss,
    reason = "clip counts are tiny — exact in f32"
)]
#[must_use]
fn as_f32(n: usize) -> f32 {
    n as f32
}

/// `n` as `f64`; analysis counts are far below 2^53.
#[expect(
    clippy::cast_precision_loss,
    reason = "analysis counts are tiny — exact in f64"
)]
#[must_use]
fn as_f64(n: usize) -> f64 {
    n as f64
}

/// Decision level of sweep index `index`: `index/1000 − 1`.
#[expect(
    clippy::cast_precision_loss,
    reason = "sweep indices are below 2001 — exact in f32"
)]
#[must_use]
fn level_of(index: usize) -> f32 {
    (index as f32 - LEVEL_ZERO_INDEX as f32) / LEVEL_SCALE
}

/// Mean of `values`, or `None` when empty.
#[must_use]
fn mean_of(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let sum: f32 = values.iter().sum();
    Some(sum / as_f32(values.len()))
}

/// Median of `values` (mean of the two middles for an even count), or `None`
/// when empty.
#[must_use]
fn median_of(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Some(sorted[middle])
    } else {
        Some(f32::midpoint(sorted[middle - 1], sorted[middle]))
    }
}

// ── Window values ────────────────────────────────────────────────────────

/// Apply a vector readout to a `tokens × dim` row-major token matrix.
///
/// Returns `None` for the scalar readouts (they compare window tokens against
/// the enrolment token pool instead of producing a window vector).
///
/// Token units: `mean` and `magnitude_weighted_mean` reduce the raw token rows
/// (the weighted mean's weights are the raw token norms, taken before any
/// normalisation); `component_max` and `last_k_*` reduce L2-normalised tokens.
///
/// The result is not L2-normalized here: callers normalize before the cosine
/// (a no-op for a cosine, which is scale-invariant).
#[must_use]
fn readout_vector(readout: Readout, matrix: &[f32], tokens: usize, dim: usize) -> Option<Vec<f32>> {
    if readout.is_scalar() || tokens == 0 || dim == 0 {
        return None;
    }
    // Entries (2) and (3) reduce normalised tokens; entries (1) and (6) reduce
    // the raw rows (the weighted mean's weights are the raw token norms).
    let mut normalized;
    let matrix = if matches!(readout, Readout::ComponentMax | Readout::LastK(_)) {
        normalized = matrix.get(..tokens * dim)?.to_vec();
        for row in normalized.chunks_exact_mut(dim) {
            l2_normalize_in_place(row);
        }
        &normalized[..]
    } else {
        matrix
    };
    let rows = matrix.chunks_exact(dim).take(tokens);
    let mut vector = vec![0.0f32; dim];
    match readout {
        Readout::Mean => {
            for row in rows {
                for (out, value) in vector.iter_mut().zip(row) {
                    *out += value;
                }
            }
            let inv = 1.0 / as_f32(tokens);
            for out in &mut vector {
                *out *= inv;
            }
        }
        Readout::ComponentMax => {
            vector.fill(f32::NEG_INFINITY);
            for row in rows {
                for (out, value) in vector.iter_mut().zip(row) {
                    if *value > *out {
                        *out = *value;
                    }
                }
            }
        }
        Readout::LastK(k) => {
            let k = usize::from(k).min(tokens);
            for row in rows.skip(tokens - k).take(k) {
                for (out, value) in vector.iter_mut().zip(row) {
                    *out += value;
                }
            }
            let inv = 1.0 / as_f32(k);
            for out in &mut vector {
                *out *= inv;
            }
        }
        Readout::MagnitudeWeightedMean => {
            let mut total_weight = 0.0f32;
            for row in rows {
                // Weight is the raw token's L2 norm — taken before any
                // normalization, so loud tokens dominate the mean.
                let weight = row.iter().map(|value| value * value).sum::<f32>().sqrt();
                total_weight += weight;
                for (out, value) in vector.iter_mut().zip(row) {
                    *out += weight * value;
                }
            }
            if total_weight > 0.0 {
                let inv = 1.0 / total_weight;
                for out in &mut vector {
                    *out *= inv;
                }
            }
        }
        Readout::TokenBestMax | Readout::TokenBestMean => return None,
    }
    Some(vector)
}

/// Per-token best cosine of one window against the whole enrolment token pool.
#[must_use]
fn token_best_cosines(
    matrix: &[f32],
    tokens: usize,
    dim: usize,
    enrollment_tokens: &[Vec<f32>],
) -> Vec<f32> {
    matrix
        .chunks_exact(dim)
        .take(tokens)
        .map(|token| {
            enrollment_tokens
                .iter()
                .map(|enrolled| cosine_similarity(token, enrolled))
                // An empty pool (no usable enrolment utterance) is the only
                // 0.0 case; the product's cosine already clamps to `[0, 1]`.
                .reduce(f32::max)
                .unwrap_or(0.0)
        })
        .collect()
}

/// The eight readout values of one window, indexed like [`Readout::ALL`].
#[must_use]
fn window_values(
    matrix: &[f32],
    tokens: usize,
    dim: usize,
    prototypes: &[Option<Vec<f32>>],
    enrollment_tokens: &[Vec<f32>],
) -> [f32; Readout::COUNT] {
    let mut values = [0.0f32; Readout::COUNT];
    let mut bests: Option<Vec<f32>> = None;
    for (index, readout) in Readout::ALL.iter().enumerate() {
        values[index] = if readout.is_scalar() {
            let per_token = bests
                .get_or_insert_with(|| token_best_cosines(matrix, tokens, dim, enrollment_tokens));
            if *readout == Readout::TokenBestMax {
                per_token.iter().copied().reduce(f32::max).unwrap_or(0.0)
            } else {
                mean_of(per_token).unwrap_or(0.0)
            }
        } else {
            let prototype = prototypes[index].as_ref().expect(
                "build_clips refuses a capture with no usable enrolment utterance, so every vector readout \
                 has a prototype",
            );
            let mut vector = readout_vector(*readout, matrix, tokens, dim)
                .expect("build_clips only evaluates non-empty windows with a vector readout");
            l2_normalize_in_place(&mut vector);
            cosine_similarity(&vector, prototype)
        };
    }
    values
}

/// The prototype for one vector readout: the same readout applied to every
/// enrolment utterance, each result L2-normalized, averaged, then normalized
/// again.  `None` for the scalar readouts.
#[must_use]
fn prototype(readout: Readout, matrices: &[(Vec<f32>, usize, usize)]) -> Option<Vec<f32>> {
    if readout.is_scalar() {
        return None;
    }
    let dim = matrices.first()?.2;
    let mut accumulator = vec![0.0f32; dim];
    let mut used = 0usize;
    for (matrix, tokens, matrix_dim) in matrices {
        if *matrix_dim != dim {
            continue;
        }
        let Some(mut vector) = readout_vector(readout, matrix, *tokens, *matrix_dim) else {
            continue;
        };
        l2_normalize_in_place(&mut vector);
        for (out, value) in accumulator.iter_mut().zip(&vector) {
            *out += value;
        }
        used += 1;
    }
    if used == 0 {
        return None;
    }
    let inv = 1.0 / as_f32(used);
    for out in &mut accumulator {
        *out *= inv;
    }
    l2_normalize_in_place(&mut accumulator);
    Some(accumulator)
}

// ── Capture records ──────────────────────────────────────────────────────

/// One `partition.jsonl` line — the frozen half assignment.
#[derive(Deserialize)]
struct PartitionLine {
    family: String,
    label: String,
    half: String,
}

/// One `clip_results_<pass>.jsonl` line.
#[derive(Deserialize)]
struct ClipResultLine {
    family: String,
    label: String,
    detected: bool,
    /// The measurement-session id the writing process carried.
    #[serde(default)]
    run: Option<String>,
}

/// One `windows.jsonl` line — one scored window, in capture order.
#[derive(Deserialize)]
struct WindowLine {
    clip: String,
    family: String,
    pos: u64,
    flush: bool,
    tokens: usize,
    dim: usize,
    offset: u64,
    product_value: f32,
}

/// One `enrollment.jsonl` line — one enrolment utterance.
#[derive(Deserialize)]
struct EnrollmentLine {
    tokens: usize,
    dim: usize,
    offset: u64,
}

/// `meta_<pass>.json` — the actual corpus sizes and integrity flags the writer
/// pinned.
#[derive(Clone, Serialize, Deserialize)]
struct MetaFile {
    /// The measurement-session id the writing process carried
    /// (`MAHBOT_WAKE_RUN`), absent when it was unset.
    #[serde(default)]
    run: Option<String>,
    /// The pass this manifest belongs to — must name the pass whose ledger it
    /// was written beside.
    #[serde(default)]
    pass: Option<String>,
    /// The selection half the writer pinned ([`SELECTION_HALF`]).
    #[serde(default)]
    selection_half: Option<String>,
    #[serde(default)]
    corpus: BTreeMap<String, Value>,
    /// The writer's report of a capture write that failed mid-run: the pass's
    /// ledger is silently short and the capture cannot be analysed.
    #[serde(default)]
    broken: Option<bool>,
    /// Windows whose token matrix was replaced before it was scored.
    #[serde(default)]
    unpaired_windows: Option<u64>,
    #[serde(default)]
    windows_captured: Option<u64>,
    #[serde(default)]
    flush_windows: Option<u64>,
    #[serde(default)]
    enrollment_utterances: Option<u64>,
}

/// Every capture artifact read from one directory.
struct Capture {
    files: Vec<FileRead>,
    tokens: Vec<u8>,
    partition: Vec<PartitionLine>,
    windows: Vec<WindowLine>,
    enrollment: Vec<EnrollmentLine>,
    clip_results: BTreeMap<String, Vec<ClipResultLine>>,
    metas: BTreeMap<String, MetaFile>,
    /// Partial trailing records skipped while parsing the JSONL files.
    partial_tails: Vec<String>,
}

/// One clip's captured windows and its frozen partition slot.
struct ClipData {
    family: String,
    label: String,
    half: String,
    /// Non-flush window values in `pos` order, one value per readout.
    values: Vec<[f32; Readout::COUNT]>,
    flush_windows: usize,
}

impl ClipData {
    fn non_flush_windows(&self) -> usize {
        self.values.len()
    }

    fn total_windows(&self) -> usize {
        self.values.len() + self.flush_windows
    }
}

/// Everything derived from the captured windows plus the enrolment matrices.
struct ClipBuild {
    clips: Vec<ClipData>,
    skipped_windows: Vec<String>,
    skipped_enrollment: Vec<String>,
    windows_without_partition: Vec<String>,
    enrollment_matrices: Vec<(Vec<f32>, usize, usize)>,
    enrollment_tokens: Vec<Vec<f32>>,
    /// Smallest value any resolved window produced across the eight readouts
    /// (`None` when no window resolved at all).
    min_window_value: Option<f32>,
    /// Bytes of `tokens.bin` and the bytes the kept records reference.
    blob_bytes: usize,
    referenced_bytes: usize,
}

// ── Report schema ────────────────────────────────────────────────────────

/// One file consumed by the analysis.
#[derive(Clone, Serialize)]
struct FileRead {
    name: String,
    lines: usize,
    bytes: usize,
    note: Option<String>,
}

/// Window/clip inventory for one group (`group` is a family or a half).
#[derive(Serialize)]
struct GroupCounts {
    group: String,
    clips: usize,
    no_window_clips: Vec<String>,
    non_flush_windows: usize,
    flush_windows: usize,
    total_windows: usize,
}

/// Enrolment corpus size.
#[derive(Serialize)]
struct EnrollmentCounts {
    utterances: usize,
    tokens: usize,
    dim: usize,
}

/// The pass metadata files — both passes always write one.
#[derive(Serialize)]
struct CorpusCounts {
    baseline: MetaFile,
    measure: MetaFile,
}

/// The measurement-session ids seen in the capture, and whether the two
/// passes could be proven to come from one session.
#[derive(Serialize)]
struct RunLinkage {
    /// Per pass: the session id in that pass's ledger.
    ledger: BTreeMap<String, Option<String>>,
    /// Per pass: the session id in that pass's manifest.
    manifest: BTreeMap<String, Option<String>>,
    /// Whether both manifests carry an id and they agree.
    asserted: bool,
    note: &'static str,
}

/// Capture integrity: what was read, what was missing, what was unusable.
#[derive(Serialize)]
struct Integrity {
    clips: usize,
    enrollment: EnrollmentCounts,
    skipped_windows: Vec<String>,
    skipped_enrollment: Vec<String>,
    windows_without_partition: Vec<String>,
    partial_tail_records: Vec<String>,
    blob_bytes: usize,
    referenced_bytes: usize,
    missing_clip_results: BTreeMap<String, Vec<String>>,
    run_linkage: RunLinkage,
    manifest_mismatches: Vec<String>,
    /// Per pass: the manifest's unpaired-window count, omitted when the
    /// manifest does not carry it.
    unpaired_windows: BTreeMap<String, u64>,
    /// Every non-empty counter above rendered as one line for an operator.
    warnings: Vec<String>,
    families: Vec<GroupCounts>,
    halves: Vec<GroupCounts>,
    corpus: CorpusCounts,
}

/// The comparison space actually used, and why half of the frozen sweep is
/// inert in it.
#[derive(Serialize)]
struct ComparisonSpace {
    similarity: &'static str,
    /// Smallest value any compared window produced across the eight readouts;
    /// the swept `[-1, 0)` half can only fire something when this is negative.
    /// `None` when no window resolved at all.
    min_window_value: Option<f32>,
    note: &'static str,
}

/// Clip counts of one half.
#[derive(Serialize)]
struct HalfCounts {
    half: String,
    role: &'static str,
    positives: usize,
    negatives: usize,
    clips: usize,
}

/// Clip counts of one family, split by half.
#[derive(Serialize)]
struct FamilyHalves {
    family: String,
    selection: usize,
    held_out: usize,
}

/// The partition as consumed.
#[derive(Serialize)]
struct PartitionReport {
    selection_half: String,
    halves: Vec<HalfCounts>,
    families: Vec<FamilyHalves>,
}

/// One family's detection counts in one pass.
#[derive(Serialize)]
struct FamilyCounts {
    family: String,
    detected: usize,
    total: usize,
}

/// One pass's per-clip ledger totals.
#[derive(Serialize)]
struct PassCounts {
    pass: String,
    positives_detected: usize,
    positives_total: usize,
    negatives_detected: usize,
    negatives_total: usize,
    families: Vec<FamilyCounts>,
}

/// The clip-by-clip difference between the two passes.
#[derive(Serialize)]
struct DifferenceReport {
    note: &'static str,
    changed_positives: Vec<String>,
    changed_negatives: Vec<String>,
    positive_changed: BTreeMap<String, usize>,
    negative_spread: BTreeMap<String, usize>,
}

/// The product's own captured per-window statistic, reported for provenance
/// only.
///
/// It is a rolling SUM in `[0, ROLLING_WINDOW_N]`, a different scale from the
/// cosine sweep in `[-1, 1]`, so it is never swept or compared: the product's
/// baseline is its own detection results, reported in [`PassCounts`].
#[derive(Serialize)]
struct ProductStatistic {
    windows: usize,
    min: f32,
    max: f32,
    mean: f32,
}

/// The two ledgers side by side.
#[derive(Serialize)]
struct BaselineReport {
    passes: Vec<PassCounts>,
    product_statistic: ProductStatistic,
    difference: DifferenceReport,
}

/// One fixed recall target mapped onto the sweep.
#[derive(Serialize)]
struct FixedLevel {
    target_recall: f32,
    /// The applied decision level — the selection half's mapping, unchanged on
    /// the held-out half.
    level: f32,
    /// The recall this half attains at `level`; on the held-out half it may sit
    /// below `target_recall`, which is the target not being attained there.
    attained_recall: f32,
    firing_negatives: usize,
}

/// One point of the compact recall-swept curve.
#[derive(Serialize)]
struct CurvePoint {
    recall: f32,
    firing_negatives: usize,
}

/// One candidate's evaluation on one half.
///
/// `recall_swept` and `auc` describe this half's OWN curve: recall is the swept
/// variable, mapped through this half's own tightest level per recall target.
/// `fixed_levels` instead carries the selection half's applied levels, so on
/// the held-out half the two readings disagree — `attained_recall` sitting below
/// `target_recall` is exactly that mismatch.
#[derive(Serialize)]
struct HalfEval {
    half: String,
    positives: usize,
    negatives: usize,
    positives_without_windows: usize,
    negatives_without_windows: usize,
    fixed_levels: Vec<FixedLevel>,
    summary_statistic: usize,
    recall_swept: Vec<CurvePoint>,
    auc: f32,
}

/// One candidate's evaluation on both halves.
#[derive(Serialize)]
struct CandidateReport {
    readout: String,
    reduction: &'static str,
    enumeration_order: usize,
    halves: Vec<HalfEval>,
}

/// The per-clip audit trail.  The trigger vectors themselves are not written
/// per clip: the analyser can recompute them from `tokens.bin`, which is what
/// keeps the analysis reproducible offline.
#[derive(Serialize)]
struct ClipReport {
    label: String,
    family: String,
    half: String,
    non_flush_windows: usize,
    flush_windows: usize,
}

/// A candidate identified by name plus the numbers a selection rests on.
#[derive(Serialize)]
struct CandidateId {
    readout: String,
    reduction: &'static str,
    summary_selection: usize,
    summary_held_out: usize,
    auc_selection: f32,
}

/// The step-2 selection result.
#[derive(Serialize)]
struct SelectionReport {
    selection_half: String,
    rule: String,
    winner: CandidateId,
    runner_up: Option<CandidateId>,
    auc_best: CandidateId,
    auc_best_disagrees: bool,
    tie_break: Option<String>,
}

/// One fixed level's held-out lift of a candidate over the control.
#[derive(Serialize)]
struct LiftPoint {
    target_recall: f32,
    control_level: f32,
    candidate_level: f32,
    lift: i64,
}

/// One candidate's verdict against the control.
#[derive(Serialize)]
struct Verdict {
    readout: String,
    reduction: &'static str,
    control_summary_selection: usize,
    candidate_summary_selection: usize,
    control_summary_held_out: usize,
    candidate_summary_held_out: usize,
    movement: i64,
    spread_held_out: usize,
    lift_held_out: Vec<LiftPoint>,
    /// Authoritative: lower firing negatives at every applied fixed level than
    /// the control, clearing the observed held-out spread.
    lifted: bool,
    rule: &'static str,
}

/// Whether one candidate could run in the streaming product as it stands.
#[derive(Serialize)]
struct AdoptabilityEntry {
    readout: String,
    reduction: &'static str,
    streaming_capable: bool,
    /// Both conditions at once: this readout can run in the streaming product as
    /// it stands, with the enrolment the product already stores.
    adoptable: bool,
    reason: String,
    /// What the product must additionally store to compare a window against
    /// this readout's prototype.  The stored [`WakeWordEnrollment`] carries the
    /// mean-readout prototype, its calibration and its anti-prototypes —
    /// nothing per utterance.
    ///
    /// [`WakeWordEnrollment`]: crate::audio::wake_word
    enrolment_data: EnrolmentData,
}

/// What the persisted enrolment must additionally carry for one readout.
#[derive(Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum EnrolmentData {
    /// The stored prototype is this readout's prototype — nothing new.
    Stored,
    /// One stored prototype per readout: the enrolled utterances reduced by
    /// this readout.  The stored enrolment carries the mean-readout prototype
    /// only.
    PerReadoutPrototype,
    /// The enrolment token matrices (or one enrolled prototype per enrolment
    /// token).  The stored enrolment carries neither token matrices nor
    /// per-utterance vectors.
    EnrolmentTokens,
}

impl EnrolmentData {
    /// The extra enrolment data one readout's prototype needs.
    #[must_use]
    const fn of(readout: Readout) -> Self {
        match readout {
            Readout::Mean => Self::Stored,
            Readout::ComponentMax | Readout::LastK(_) | Readout::MagnitudeWeightedMean => {
                Self::PerReadoutPrototype
            }
            Readout::TokenBestMax | Readout::TokenBestMean => Self::EnrolmentTokens,
        }
    }
}

/// Adoptability of the closed candidate list.
///
/// `adoptable as it stands` requires both facts: the reduction streams, and the
/// product already stores what scoring the readout needs.
#[derive(Serialize)]
struct Adoptability {
    comparison_space_excludes: Vec<String>,
    candidates: Vec<AdoptabilityEntry>,
}

/// The full analysis, serialized verbatim to `analysis.json`.
#[derive(Serialize)]
struct Report {
    capture_dir: String,
    files_read: Vec<FileRead>,
    timestamp: String,
    integrity: Integrity,
    comparison_space: ComparisonSpace,
    partition: PartitionReport,
    baseline: BaselineReport,
    candidates: Vec<CandidateReport>,
    per_clip: Vec<ClipReport>,
    selection: SelectionReport,
    verdicts: Vec<Verdict>,
    adoptability: Adoptability,
    caveats: Vec<String>,
}

// ── Reading ──────────────────────────────────────────────────────────────

/// View one captured token matrix as `f32`s, together with the byte range the
/// record names in the blob, or `None` when the record is unusable (zero
/// tokens/dim, or a byte range outside the captured blob).
#[must_use]
fn resolve_record(
    blob: &[u8],
    offset: u64,
    tokens: usize,
    dim: usize,
) -> Option<(Range<usize>, Vec<f32>)> {
    if tokens == 0 || dim == 0 {
        return None;
    }
    let byte_len = tokens.checked_mul(dim)?.checked_mul(4)?;
    let start = usize::try_from(offset).ok()?;
    let end = start.checked_add(byte_len)?;
    let bytes = blob.get(start..end)?;
    let (chunks, _remainder) = bytes.as_chunks::<4>();
    let matrix = chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    Some((start..end, matrix))
}

/// Parse the non-empty lines of one JSONL capture file.
///
/// The writer appends straight to the file without buffering, so a process
/// killed mid-record leaves a partial LAST line: that line alone is skipped and
/// recorded in `partial_tails`.  A line that fails to parse anywhere else is a
/// hard error — it cannot be the tail the writer's design leaves behind.
fn parse_jsonl<T: DeserializeOwned>(
    name: &str,
    text: &str,
    partial_tails: &mut Vec<String>,
) -> Result<Vec<T>> {
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    let mut records = Vec::with_capacity(lines.len());
    for (index, line) in lines.iter().enumerate() {
        match serde_json::from_str(line) {
            Ok(record) => records.push(record),
            Err(_) if index + 1 == lines.len() => partial_tails.push(format!(
                "{name}: line {} is a partial trailing record and was skipped",
                index + 1
            )),
            Err(error) => {
                return Err(anyhow::Error::new(error)
                    .context(format!("{name}: line {} is not a valid record", index + 1)));
            }
        }
    }
    Ok(records)
}

/// Read a required text capture file (recording it for the report).
fn required_text(dir: &Path, name: &str, files: &mut Vec<FileRead>) -> Result<String> {
    match fs::read_to_string(dir.join(name)) {
        Ok(text) => {
            files.push(FileRead {
                name: name.to_string(),
                lines: text.lines().filter(|line| !line.trim().is_empty()).count(),
                bytes: text.len(),
                note: None,
            });
            Ok(text)
        }
        Err(error) => Err(anyhow!(
            "{name} is missing or unreadable in {}: {error}",
            dir.display()
        )),
    }
}

/// Read the required token blob (recording it for the report).
fn required_tokens(dir: &Path, files: &mut Vec<FileRead>) -> Result<Vec<u8>> {
    match fs::read(dir.join(FILE_TOKENS)) {
        Ok(bytes) => {
            files.push(FileRead {
                name: FILE_TOKENS.to_string(),
                lines: 0,
                bytes: bytes.len(),
                note: Some("little-endian f32 token blobs".to_string()),
            });
            Ok(bytes)
        }
        Err(error) => Err(anyhow!(
            "{FILE_TOKENS} is missing or unreadable in {}: {error}",
            dir.display()
        )),
    }
}

/// Read one required `meta_<pass>.json` — both passes always write a manifest,
/// so a missing or unreadable one is a hard error.
fn required_meta(dir: &Path, pass: &str, files: &mut Vec<FileRead>) -> Result<MetaFile> {
    let name = meta_file(pass);
    let text = required_text(dir, &name, files)?;
    serde_json::from_str(&text).with_context(|| format!("{name} is not a valid metadata file"))
}

/// Read every capture artifact.  A missing required file is a hard error: the
/// analyser never guesses around an incomplete capture.
fn load(dir: &Path) -> Result<Capture> {
    let mut files = Vec::new();
    let mut partial_tails = Vec::new();
    let partition_text = required_text(dir, FILE_PARTITION, &mut files)?;
    let windows_text = required_text(dir, FILE_WINDOWS, &mut files)?;
    let enrollment_text = required_text(dir, FILE_ENROLLMENT, &mut files)?;
    let tokens = required_tokens(dir, &mut files)?;

    let mut clip_results = BTreeMap::new();
    for pass in PASSES {
        let name = clip_results_file(pass);
        let text = required_text(dir, name, &mut files)?;
        clip_results.insert(
            pass.to_string(),
            parse_jsonl(name, &text, &mut partial_tails)?,
        );
    }
    let mut metas = BTreeMap::new();
    for pass in PASSES {
        metas.insert(pass.to_string(), required_meta(dir, pass, &mut files)?);
    }

    let partition = parse_jsonl(FILE_PARTITION, &partition_text, &mut partial_tails)?;
    let windows = parse_jsonl(FILE_WINDOWS, &windows_text, &mut partial_tails)?;
    let enrollment = parse_jsonl(FILE_ENROLLMENT, &enrollment_text, &mut partial_tails)?;

    Ok(Capture {
        files,
        tokens,
        partition,
        windows,
        enrollment,
        clip_results,
        metas,
        partial_tails,
    })
}

// ── Window derivation ────────────────────────────────────────────────────

/// Turn the captured token blob into clip-attached window values.
///
/// # Errors
/// A capture with no usable enrolment utterance: every vector readout would
/// then have no prototype and would score every window as 0.
fn build_clips(capture: &Capture) -> Result<ClipBuild> {
    let mut enrollment_matrices = Vec::new();
    let mut enrollment_tokens = Vec::new();
    let mut skipped_enrollment = Vec::new();
    // The bytes of the blob the resolved records reference — the blob's own
    // accounting, checked at the end.
    let mut referenced_bytes = 0usize;
    for record in &capture.enrollment {
        match resolve_record(&capture.tokens, record.offset, record.tokens, record.dim) {
            Some((range, matrix)) => {
                referenced_bytes += range.end - range.start;
                for token in matrix.chunks_exact(record.dim) {
                    enrollment_tokens.push(token.to_vec());
                }
                enrollment_matrices.push((matrix, record.tokens, record.dim));
            }
            None => skipped_enrollment.push(format!(
                "offset {} tokens {} dim {}",
                record.offset, record.tokens, record.dim
            )),
        }
    }
    anyhow::ensure!(
        !enrollment_matrices.is_empty(),
        "{FILE_ENROLLMENT} holds no usable enrolment utterance — every vector readout would score 0"
    );

    let prototypes: [Option<Vec<f32>>; Readout::COUNT] =
        std::array::from_fn(|index| prototype(Readout::ALL[index], &enrollment_matrices));

    // Partition first, so clips keep the frozen order and half assignment.
    let mut clips: Vec<ClipData> = Vec::new();
    let mut lookup: BTreeMap<(String, String), usize> = BTreeMap::new();
    for entry in &capture.partition {
        let key = (entry.family.clone(), entry.label.clone());
        if lookup.contains_key(&key) {
            continue;
        }
        lookup.insert(key, clips.len());
        clips.push(ClipData {
            family: entry.family.clone(),
            label: entry.label.clone(),
            half: entry.half.clone(),
            values: Vec::new(),
            flush_windows: 0,
        });
    }

    let mut pending: Vec<Vec<(u64, bool, [f32; Readout::COUNT])>> = vec![Vec::new(); clips.len()];
    let mut skipped_windows = Vec::new();
    let mut min_window_value: Option<f32> = None;
    let mut outside = BTreeSet::new();
    for record in &capture.windows {
        let key = (record.family.clone(), record.clip.clone());
        let Some(&index) = lookup.get(&key) else {
            outside.insert(format!("{} / {}", record.family, record.clip));
            continue;
        };
        // An empty window has no matrix to reduce and would score as nothing.
        if record.tokens == 0 || record.dim == 0 {
            skipped_windows.push(format!("{} pos {} has no matrix", record.clip, record.pos));
            continue;
        }
        match resolve_record(&capture.tokens, record.offset, record.tokens, record.dim) {
            Some((range, matrix)) => {
                referenced_bytes += range.end - range.start;
                let values = window_values(
                    &matrix,
                    record.tokens,
                    record.dim,
                    &prototypes,
                    &enrollment_tokens,
                );
                min_window_value = values
                    .iter()
                    .copied()
                    .chain(min_window_value)
                    .reduce(f32::min);
                pending[index].push((record.pos, record.flush, values));
            }
            None => skipped_windows.push(format!(
                "{} pos {} offset {} tokens {} dim {}",
                record.clip, record.pos, record.offset, record.tokens, record.dim
            )),
        }
    }

    for (clip, mut rows) in clips.iter_mut().zip(pending) {
        rows.sort_by_key(|(pos, _, _)| *pos);
        for (_, flush, values) in rows {
            if flush {
                clip.flush_windows += 1;
            } else {
                clip.values.push(values);
            }
        }
    }

    Ok(ClipBuild {
        clips,
        skipped_windows,
        skipped_enrollment,
        windows_without_partition: outside.into_iter().collect(),
        enrollment_matrices,
        enrollment_tokens,
        min_window_value,
        blob_bytes: capture.tokens.len(),
        referenced_bytes,
    })
}

/// One candidate's trigger values for every clip (aligned with
/// [`ClipBuild::clips`]).
#[must_use]
fn triggers_for_candidate(candidate: Candidate, clips: &[ClipData]) -> Vec<Vec<f32>> {
    let index = candidate.readout.index();
    clips
        .iter()
        .map(|clip| {
            let values: Vec<f32> = clip.values.iter().map(|window| window[index]).collect();
            triggers(candidate.reduction, &values)
        })
        .collect()
}

// ── Sweep and evaluation ─────────────────────────────────────────────────

/// Index of the tightest (highest) swept level that attains `target` recall.
///
/// The mapping is "the level that attains the recall level", read as the
/// tightest such level: recall is monotone non-increasing in `L`, so the
/// literal "lowest swept level attaining `target`" would always be `L = −1.0`
/// and all six targets would collapse to one working point where every
/// candidate's false-accept count is identical — the comparison would be
/// vacuous.  The tightest level is the only non-degenerate reading; the
/// applied-level convention built on it is stated in the module doc.
#[must_use]
fn tightest_level(recall: &[f32], target: f32) -> usize {
    (0..recall.len())
        .rev()
        .find(|&index| recall[index] >= target)
        .unwrap_or(0)
}

/// The firing-negative count at the highest fixed recall level this half
/// attains — the step-2 first tie-break.  `None` when the half attains none of
/// the fixed targets, which takes more than half of the half's positives to
/// have no scored window at all (the loosest target, R = 0.50, is already
/// attained by every clip that scores a window).
///
/// `None` ranks strictly last: a candidate that attains at least one target is
/// compared at the tightest target it reaches, while one that attains nothing
/// has no such comparison to offer and loses the tie-break.
#[must_use]
fn top_level_fn(half: &HalfEval) -> Option<usize> {
    half.fixed_levels
        .iter()
        .find(|level| level.attained_recall >= level.target_recall)
        .map(|level| level.firing_negatives)
}

/// The step-2 first tie-break's ordering: `None` last, `(None, None)` equal.
///
/// This is not `Option::cmp`: that sorts `None` first, which would promote the
/// candidate that attains no fixed target at all.
#[must_use]
fn top_level_cmp(a: &HalfEval, b: &HalfEval) -> std::cmp::Ordering {
    match (top_level_fn(a), top_level_fn(b)) {
        (Some(a), Some(b)) => a.cmp(&b),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

/// One half's swept recall-versus-firing-negatives curve, before any fixed
/// level is mapped.
struct HalfSweep {
    half: String,
    positives: usize,
    negatives: usize,
    positives_without_windows: usize,
    negatives_without_windows: usize,
    /// Recall at each swept level (`LEVEL_STEPS` entries).
    recall: Vec<f32>,
    /// Firing negatives at each swept level.
    negatives_firing: Vec<usize>,
}

/// Sweep one candidate's trigger values over one half.
///
/// Clips with no non-flush windows get `−∞`: they fire at no swept level, but
/// they stay in the denominator (a positive without windows is a recall miss).
#[must_use]
fn sweep_half(half: &str, clips: &[ClipData], candidate_triggers: &[Vec<f32>]) -> HalfSweep {
    let mut positives: Vec<f32> = Vec::new();
    let mut negatives: Vec<f32> = Vec::new();
    for (clip, trigger) in clips.iter().zip(candidate_triggers) {
        if clip.half != half {
            continue;
        }
        let value = trigger
            .iter()
            .copied()
            .reduce(f32::max)
            .unwrap_or(f32::NEG_INFINITY);
        if clip.family == POSITIVE_FAMILY {
            positives.push(value);
        } else {
            negatives.push(value);
        }
    }

    let mut recall = vec![0.0f32; LEVEL_STEPS];
    let mut negatives_firing = vec![0usize; LEVEL_STEPS];
    let positive_total = positives.len();
    for (index, level) in (0..LEVEL_STEPS).map(level_of).enumerate() {
        let hits = positives.iter().filter(|value| **value >= level).count();
        recall[index] = ratio(hits, positive_total);
        negatives_firing[index] = negatives.iter().filter(|value| **value >= level).count();
    }
    // A window-less clip carries −∞, so `is_finite()` records exactly the clips
    // that can never fire on this half.
    let without_windows = |values: &[f32]| values.iter().filter(|value| !value.is_finite()).count();

    HalfSweep {
        half: half.to_string(),
        positives: positives.len(),
        negatives: negatives.len(),
        positives_without_windows: without_windows(&positives),
        negatives_without_windows: without_windows(&negatives),
        recall,
        negatives_firing,
    }
}

/// The six decision levels a half's own curve maps: the tightest (highest)
/// swept level attaining each target, with the most permissive level as the
/// fallback for an unattainable target.
#[must_use]
fn own_level_indices(sweep: &HalfSweep) -> Vec<usize> {
    FIXED_RECALLS
        .iter()
        .map(|&target| tightest_level(&sweep.recall, target))
        .collect()
}

/// Evaluate one candidate on every half.  The decision levels are built once on
/// the selection half and applied unchanged to every half — including the
/// selection half itself, where the applied levels are its own mapping.
#[must_use]
fn evaluate_candidate_halves(
    halves: &[String],
    selection: usize,
    clips: &[ClipData],
    candidate_triggers: &[Vec<f32>],
) -> Vec<HalfEval> {
    let sweeps: Vec<HalfSweep> = halves
        .iter()
        .map(|half| sweep_half(half, clips, candidate_triggers))
        .collect();
    let applied = own_level_indices(&sweeps[selection]);
    sweeps
        .into_iter()
        .map(|sweep| evaluate_sweep(sweep, &applied))
        .collect()
}

/// Turn one half's sweep into its evaluated cells at the applied decision
/// levels.
///
/// The authoritative cell for target `R` is `(level, firing_negatives,
/// attained_recall)` at the applied level — on the held-out half that level is
/// the selection half's, so `attained_recall` may sit below `R` and the
/// shortfall is reported, never silently corrected.
#[must_use]
fn evaluate_sweep(sweep: HalfSweep, applied: &[usize]) -> HalfEval {
    let mut fixed_levels = Vec::with_capacity(FIXED_RECALLS.len());
    let mut summary_statistic = 0usize;
    for (slot, &target) in FIXED_RECALLS.iter().enumerate() {
        let index = applied[slot];
        let attained_recall = sweep.recall[index];
        let firing_negatives = sweep.negatives_firing[index];
        summary_statistic += firing_negatives;
        fixed_levels.push(FixedLevel {
            target_recall: target,
            level: level_of(index),
            attained_recall,
            firing_negatives,
        });
    }

    let mut recall_swept = Vec::with_capacity(CURVE_GRID_STEPS);
    for step in 1..=CURVE_GRID_STEPS {
        let target = as_f32(step) / as_f32(CURVE_GRID_STEPS);
        recall_swept.push(CurvePoint {
            recall: target,
            firing_negatives: sweep.negatives_firing[tightest_level(&sweep.recall, target)],
        });
    }

    // AUC of the recall-versus-false-accepts curve, with recall as the swept
    // variable and the same tightest-level mapping (trapezoid rule).
    let grid: Vec<f64> = (0..=AUC_GRID_STEPS)
        .map(|step| {
            let target = as_f32(step) / as_f32(AUC_GRID_STEPS);
            as_f64(sweep.negatives_firing[tightest_level(&sweep.recall, target)])
        })
        .collect();
    let mut auc = 0.0f64;
    let step_width = 1.0 / as_f64(AUC_GRID_STEPS);
    for pair in grid.windows(2) {
        auc += f64::midpoint(pair[0], pair[1]) * step_width;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "AUC is rounded to 6 decimals before the f32 narrowing"
    )]
    let auc = (auc * 1e6).round() as f32 / 1e6;

    HalfEval {
        half: sweep.half,
        positives: sweep.positives,
        negatives: sweep.negatives,
        positives_without_windows: sweep.positives_without_windows,
        negatives_without_windows: sweep.negatives_without_windows,
        fixed_levels,
        summary_statistic,
        recall_swept,
        auc,
    }
}

/// The step-2 selection ordering on one half: summary statistic, then the
/// firing-negative count at the highest attainable fixed recall level, then
/// AUC, then the fixed enumeration order.
#[must_use]
fn selection_cmp(a: &HalfEval, b: &HalfEval, order_a: usize, order_b: usize) -> std::cmp::Ordering {
    a.summary_statistic
        .cmp(&b.summary_statistic)
        .then_with(|| top_level_cmp(a, b))
        .then_with(|| {
            a.auc
                .partial_cmp(&b.auc)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| order_a.cmp(&order_b))
}

/// Which step of [`selection_cmp`] separated two candidates.
#[must_use]
fn tie_break_reason(a: &HalfEval, b: &HalfEval) -> &'static str {
    if a.summary_statistic != b.summary_statistic {
        "summary statistic"
    } else if top_level_fn(a) != top_level_fn(b) {
        "firing negatives at the highest attainable fixed recall level"
    } else if a.auc.partial_cmp(&b.auc) != Some(std::cmp::Ordering::Equal) {
        "AUC"
    } else {
        "fixed enumeration order"
    }
}

/// The verdict rule precedence.
///
/// Rule one is checked first because `lifted` requires all six per-level lifts
/// to beat the observed spread, so a lifted candidate moves at least six counts
/// — outside rule one's resolution and strictly better than the control.  That
/// also makes rule three's and rule four's `strictly_worse` guards unreachable
/// from the lifted branch, so the precedence is structural, not a choice.
#[must_use]
fn verdict_rule(movement: i64, selection_advantage: bool, lifted: bool) -> &'static str {
    let strictly_worse = movement < 0;
    if movement.abs() <= MOVEMENT_RESOLUTION {
        "rule one (within resolution)"
    } else if lifted {
        "separation lifted"
    } else if selection_advantage && strictly_worse {
        "rule three (advantage on the selection half reverses on the held-out half)"
    } else if strictly_worse {
        "rule four (strictly worse on the held-out half)"
    } else {
        "rule two (same recall-versus-false-accepts exchange, separation not lifted)"
    }
}

// ── Report assembly ──────────────────────────────────────────────────────

/// Half names in report order: the selection half first, then the rest.
#[must_use]
fn half_names(capture: &Capture) -> Vec<String> {
    let mut halves: BTreeSet<String> = capture
        .partition
        .iter()
        .map(|entry| entry.half.clone())
        .collect();
    let mut ordered = Vec::new();
    if halves.remove(SELECTION_HALF) {
        ordered.push(SELECTION_HALF.to_string());
    }
    ordered.extend(halves);
    ordered
}

/// Window/clip inventory grouped by family and by half.
#[must_use]
fn group_counts(build: &ClipBuild, halves: &[String]) -> (Vec<GroupCounts>, Vec<GroupCounts>) {
    let mut by_family: BTreeMap<String, Vec<&ClipData>> = BTreeMap::new();
    for clip in &build.clips {
        by_family.entry(clip.family.clone()).or_default().push(clip);
    }
    let families = by_family
        .into_iter()
        .map(|(group, clips)| summarize(group, &clips))
        .collect();

    let halves = halves
        .iter()
        .map(|half| {
            let clips: Vec<&ClipData> = build
                .clips
                .iter()
                .filter(|clip| clip.half == *half)
                .collect();
            summarize(half.clone(), &clips)
        })
        .collect();
    (families, halves)
}

/// Reduce one group's clips to its inventory row.
#[must_use]
fn summarize(group: String, clips: &[&ClipData]) -> GroupCounts {
    GroupCounts {
        group,
        clips: clips.len(),
        no_window_clips: clips
            .iter()
            .filter(|clip| clip.total_windows() == 0)
            .map(|clip| clip.label.clone())
            .collect(),
        non_flush_windows: clips.iter().map(|clip| clip.non_flush_windows()).sum(),
        flush_windows: clips.iter().map(|clip| clip.flush_windows).sum(),
        total_windows: clips.iter().map(|clip| clip.total_windows()).sum(),
    }
}

/// Clip labels present in the partition but missing from one pass ledger.
#[must_use]
fn missing_clip_results(capture: &Capture, pass: &str) -> Vec<String> {
    let present: BTreeSet<(String, String)> = capture.clip_results[pass]
        .iter()
        .map(|record| (record.family.clone(), record.label.clone()))
        .collect();
    capture
        .partition
        .iter()
        .filter(|entry| !present.contains(&(entry.family.clone(), entry.label.clone())))
        .map(|entry| format!("{} / {}", entry.family, entry.label))
        .collect()
}

/// The session ids the capture carries, and whether the two passes can be
/// proven to come from one measurement session.
///
/// The passes are two processes of one session: a pass's ledger and its
/// manifest are written by that same process, so their ids must agree, and both
/// manifests must agree with each other.  `MAHBOT_WAKE_RUN` is what makes the
/// identity checkable at all; an unset id leaves the linkage unasserted but
/// still cross-checked, so a mixed capture is caught rather than analysed.
///
/// # Errors
/// A pass whose ledger carries two distinct ids, a pass whose ledger and
/// manifest disagree, two manifests that disagree, or exactly one manifest
/// carrying an id: all four mean the directory holds artefacts of more than one
/// session and every number derived from it would be meaningless.
fn run_linkage(capture: &Capture) -> Result<RunLinkage> {
    let mut ledger: BTreeMap<String, Option<String>> = BTreeMap::new();
    let mut manifest: BTreeMap<String, Option<String>> = BTreeMap::new();
    for pass in PASSES {
        let ids: BTreeSet<&str> = capture.clip_results[pass]
            .iter()
            .filter_map(|record| record.run.as_deref())
            .collect();
        if ids.len() > 1 {
            return Err(anyhow!(
                "{} carries {} distinct run ids ({}) — a pass writes one ledger, so it cannot hold two \
                 sessions",
                clip_results_file(pass),
                ids.len(),
                ids.into_iter().collect::<Vec<_>>().join(", ")
            ));
        }
        ledger.insert(pass.to_string(), ids.into_iter().next().map(str::to_string));
        manifest.insert(pass.to_string(), capture.metas[pass].run.clone());
    }

    for pass in PASSES {
        if let (Some(ledger_id), Some(manifest_id)) = (&ledger[pass], &manifest[pass])
            && ledger_id != manifest_id
        {
            return Err(anyhow!(
                "{pass}: {} records run {ledger_id} but {} records run {manifest_id} — one process writes \
                 both",
                clip_results_file(pass),
                meta_file(pass)
            ));
        }
    }

    match (&manifest[PASS_BASELINE], &manifest[PASS_MEASURE]) {
        (Some(baseline), Some(measure)) if baseline != measure => {
            return Err(anyhow!(
                "{} records run {baseline} but {} records run {measure} — the two passes must be one \
                 measurement session",
                meta_file(PASS_BASELINE),
                meta_file(PASS_MEASURE)
            ));
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err(anyhow!(
                "exactly one manifest carries a run id — MAHBOT_WAKE_RUN must be set for both passes or \
                 neither"
            ));
        }
        _ => {}
    }

    let asserted = manifest.values().all(Option::is_some);
    Ok(RunLinkage {
        ledger,
        manifest,
        asserted,
        note: if asserted {
            "both manifests carry the same MAHBOT_WAKE_RUN id, so the two passes are one measurement session"
        } else {
            "MAHBOT_WAKE_RUN was unset, so the two passes cannot be proven to come from one session — the \
             linkage is unasserted for this capture"
        },
    })
}

/// The passes whose manifest reports a capture write that failed mid-run.
///
/// A failed write ends capture for the rest of that process, so the pass's
/// ledger stops short and everything derived from it would be silently
/// incomplete — this is the one integrity flag that makes the analysis
/// impossible rather than merely qualified.
#[must_use]
fn broken_manifests(capture: &Capture) -> Vec<String> {
    PASSES
        .iter()
        .filter(|pass| capture.metas[**pass].broken == Some(true))
        .map(|pass| (*pass).to_string())
        .collect()
}

/// The convention every manifest must be written under.
///
/// Both are pinned by the writer: the manifest names its own pass and the
/// frozen selection half.  A manifest that names another pass, another
/// selection half, or nothing at all was left behind by a different convention
/// and none of its counts describe this capture.
///
/// # Errors
/// A manifest that does not name its own pass or does not name
/// [`SELECTION_HALF`].
fn manifest_conventions(capture: &Capture) -> Result<()> {
    for pass in PASSES {
        let meta = &capture.metas[pass];
        if meta.pass.as_deref() != Some(pass) {
            return Err(anyhow!(
                "{} records pass {} — a manifest must name the pass whose ledger it sits beside",
                meta_file(pass),
                render_optional(meta.pass.as_deref())
            ));
        }
        if meta.selection_half.as_deref() != Some(SELECTION_HALF) {
            return Err(anyhow!(
                "{} records selection half {} — every manifest must record the frozen {SELECTION_HALF}",
                meta_file(pass),
                render_optional(meta.selection_half.as_deref())
            ));
        }
    }
    Ok(())
}

/// The manifests against the artefacts they name — reported, never fatal and
/// never corrected.
///
/// A mismatch means the manifest does not describe the directory it sits in:
/// the ledger it was written next to has been replaced, or the window and
/// enrolment files were truncated or appended to since.
#[must_use]
fn manifest_mismatches(capture: &Capture) -> Vec<String> {
    let mut mismatches = Vec::new();

    // Only the measure pass captures windows and enrolment audio, so only its
    // manifest carries these counts; the baseline manifest is exempt.
    let measure = &capture.metas[PASS_MEASURE];
    let flush = capture.windows.iter().filter(|record| record.flush).count();
    for (field, manifest, captured) in [
        (
            "windows_captured",
            measure.windows_captured,
            capture.windows.len(),
        ),
        ("flush_windows", measure.flush_windows, flush),
        (
            "enrollment_utterances",
            measure.enrollment_utterances,
            capture.enrollment.len(),
        ),
    ] {
        if let Some(mismatch) = count_mismatch(PASS_MEASURE, field, manifest, Some(captured)) {
            mismatches.push(mismatch);
        }
    }

    for pass in PASSES {
        let mut totals: BTreeMap<&str, usize> = BTreeMap::new();
        for record in &capture.clip_results[pass] {
            *totals.entry(record.family.as_str()).or_default() += 1;
        }
        let corpus = &capture.metas[pass].corpus;
        let families: BTreeSet<&str> = totals
            .keys()
            .copied()
            .chain(corpus.keys().map(String::as_str))
            .collect();
        for family in families {
            let manifest = corpus.get(family).and_then(Value::as_u64);
            if let Some(mismatch) =
                count_mismatch(pass, family, manifest, totals.get(family).copied())
            {
                mismatches.push(mismatch);
            }
        }
    }

    mismatches
}

/// One manifest number that disagrees with the capture, or `None` when they
/// agree.  `captured` is `None` when the capture holds no such entry at all.
#[must_use]
fn count_mismatch(
    pass: &str,
    field: &str,
    manifest: Option<u64>,
    captured: Option<usize>,
) -> Option<String> {
    if manifest == captured.and_then(|captured| u64::try_from(captured).ok()) {
        return None;
    }
    Some(format!(
        "{pass}: {field} — manifest {}, capture {}",
        render_optional(manifest),
        render_optional(captured)
    ))
}

/// An optional manifest value as the report prints it (`absent` when the
/// manifest does not carry it).
#[must_use]
fn render_optional<T: ToString>(value: Option<T>) -> String {
    value.map_or_else(|| "absent".to_string(), |value| value.to_string())
}

/// Every non-empty integrity counter as one operator-facing line.
#[must_use]
fn integrity_warnings(integrity: &Integrity) -> Vec<String> {
    let mut warnings = Vec::new();
    for (label, count) in [
        ("skipped windows", integrity.skipped_windows.len()),
        (
            "skipped enrolment records",
            integrity.skipped_enrollment.len(),
        ),
        (
            "windows without a partition entry",
            integrity.windows_without_partition.len(),
        ),
        (
            "partial trailing records",
            integrity.partial_tail_records.len(),
        ),
        ("manifest mismatches", integrity.manifest_mismatches.len()),
    ] {
        if count > 0 {
            warnings.push(format!(
                "{count} {label} — see integrity in {ANALYSIS_JSON}"
            ));
        }
    }
    for (pass, clips) in &integrity.missing_clip_results {
        if !clips.is_empty() {
            warnings.push(format!(
                "{pass}: {} clips have no ledger entry",
                clips.len()
            ));
        }
    }
    for (pass, count) in &integrity.unpaired_windows {
        if *count > 0 {
            warnings.push(format!(
                "{pass}: {count} windows had their token matrix replaced before scoring"
            ));
        }
    }
    if integrity.blob_bytes != integrity.referenced_bytes {
        warnings.push(format!(
            "{FILE_TOKENS} holds {} bytes but the kept records reference {}",
            integrity.blob_bytes, integrity.referenced_bytes
        ));
    }
    warnings
}

/// The partition as consumed: half assignment and per-family split.
#[must_use]
fn partition_report(capture: &Capture, halves: &[String]) -> PartitionReport {
    let mut counts: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    let mut families: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for entry in &capture.partition {
        let slot = counts.entry(entry.half.clone()).or_default();
        slot.2 += 1;
        if entry.family == POSITIVE_FAMILY {
            slot.0 += 1;
        } else {
            slot.1 += 1;
        }
        let family = families.entry(entry.family.clone()).or_default();
        if entry.half == SELECTION_HALF {
            family.0 += 1;
        } else {
            family.1 += 1;
        }
    }
    PartitionReport {
        selection_half: SELECTION_HALF.to_string(),
        halves: halves
            .iter()
            .map(|half| {
                let (positives, negatives, clips) = counts.get(half).copied().unwrap_or_default();
                HalfCounts {
                    half: half.clone(),
                    role: if half == SELECTION_HALF {
                        "selection"
                    } else {
                        "held-out"
                    },
                    positives,
                    negatives,
                    clips,
                }
            })
            .collect(),
        families: families
            .into_iter()
            .map(|(family, (selection, held_out))| FamilyHalves {
                family,
                selection,
                held_out,
            })
            .collect(),
    }
}

/// One pass's ledger totals.
#[must_use]
fn pass_counts(pass: &str, records: &[ClipResultLine]) -> PassCounts {
    let mut families: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let (mut positives_detected, mut positives_total) = (0usize, 0usize);
    let (mut negatives_detected, mut negatives_total) = (0usize, 0usize);
    for record in records {
        let family = families.entry(record.family.clone()).or_default();
        family.1 += 1;
        if record.detected {
            family.0 += 1;
        }
        if record.family == POSITIVE_FAMILY {
            positives_total += 1;
            if record.detected {
                positives_detected += 1;
            }
        } else {
            negatives_total += 1;
            if record.detected {
                negatives_detected += 1;
            }
        }
    }
    PassCounts {
        pass: pass.to_string(),
        positives_detected,
        positives_total,
        negatives_detected,
        negatives_total,
        families: families
            .into_iter()
            .map(|(family, (detected, total))| FamilyCounts {
                family,
                detected,
                total,
            })
            .collect(),
    }
}

/// The two ledgers side by side plus the clip-by-clip difference.
#[must_use]
fn baseline_report(capture: &Capture) -> BaselineReport {
    let detections: Vec<BTreeMap<(String, String), bool>> = PASSES
        .iter()
        .map(|pass| {
            capture.clip_results[*pass]
                .iter()
                .map(|record| {
                    (
                        (record.family.clone(), record.label.clone()),
                        record.detected,
                    )
                })
                .collect()
        })
        .collect();

    let mut changed_positives = Vec::new();
    let mut changed_negatives = Vec::new();
    let mut positive_changed: BTreeMap<String, usize> = BTreeMap::new();
    let mut negative_spread: BTreeMap<String, usize> = BTreeMap::new();
    for entry in &capture.partition {
        let key = (entry.family.clone(), entry.label.clone());
        let (Some(baseline), Some(measure)) = (detections[0].get(&key), detections[1].get(&key))
        else {
            continue;
        };
        if baseline == measure {
            continue;
        }
        if entry.family == POSITIVE_FAMILY {
            changed_positives.push(entry.label.clone());
            *positive_changed.entry(entry.half.clone()).or_default() += 1;
        } else {
            changed_negatives.push(entry.label.clone());
            *negative_spread.entry(entry.half.clone()).or_default() += 1;
        }
    }

    BaselineReport {
        passes: PASSES
            .iter()
            .map(|pass| pass_counts(pass, &capture.clip_results[*pass]))
            .collect(),
        product_statistic: product_statistic(capture),
        difference: DifferenceReport {
            note: "the observed difference between the two passes, NOT run-to-run nondeterminism: the measured \
                   pass deliberately keeps feeding every clip instead of stopping at the first fire, so the \
                   shipped post-fire suppression is neutralised",
            changed_positives,
            changed_negatives,
            positive_changed,
            negative_spread,
        },
    }
}

/// The captured product statistic's own summary (provenance only).
#[must_use]
fn product_statistic(capture: &Capture) -> ProductStatistic {
    let values: Vec<f32> = capture
        .windows
        .iter()
        .map(|line| line.product_value)
        .collect();
    let mean = values.iter().sum::<f32>() / as_f32(values.len().max(1));
    ProductStatistic {
        windows: values.len(),
        min: values.iter().copied().reduce(f32::min).unwrap_or(0.0),
        max: values.iter().copied().reduce(f32::max).unwrap_or(0.0),
        mean,
    }
}

/// The caveats the report always carries.
#[must_use]
fn caveats() -> Vec<String> {
    vec![
        "the effective negative denominator can sit below 113: structurally scoreless clips (digital silence, \
         noise profiles that never pass the VAD) score no window at all, so they count as a no-fire for every \
         candidate and every level."
            .to_string(),
        "the comparison space deliberately excludes the product's calibration floor, anti-prototype veto and \
         adaptive threshold, so a win over the control here does not transfer to the shipped rule without being \
         re-measured there."
            .to_string(),
        "the two halves are close variants of the same material (paired renditions of one voice and one positive \
         phrase), so the held-out half is a stability check, not independent evidence."
            .to_string(),
    ]
}

/// Adoptability of every computed candidate.
#[must_use]
fn adoptability(candidates: &[Candidate]) -> Adoptability {
    Adoptability {
        comparison_space_excludes: [
            "calibration floor",
            "anti-prototype veto",
            "adaptive threshold",
        ]
        .iter()
        .map(|item| (*item).to_string())
        .collect(),
        candidates: candidates
            .iter()
            .map(|candidate| {
                let streaming_capable = candidate.reduction == Reduction::Sliding3;
                let enrolment_data = EnrolmentData::of(candidate.readout);
                AdoptabilityEntry {
                    readout: candidate.readout.name(),
                    reduction: candidate.reduction.name(),
                    streaming_capable,
                    adoptable: streaming_capable && enrolment_data == EnrolmentData::Stored,
                    reason: if streaming_capable {
                        "a sliding mean of three consecutive windows needs no look-ahead"
                            .to_string()
                    } else {
                        format!(
                            "{} needs the whole clip before it can be reduced",
                            candidate.reduction.name()
                        )
                    },
                    enrolment_data,
                }
            })
            .collect(),
    }
}

/// Read the capture directory and compute the whole analysis.
#[expect(
    clippy::too_many_lines,
    reason = "linear report assembly, kept in one place so the JSON key order stays visible"
)]
fn analyse(dir: &Path) -> Result<Report> {
    let capture = load(dir)?;
    manifest_conventions(&capture)?;
    let broken = broken_manifests(&capture);
    if !broken.is_empty() {
        return Err(anyhow!(
            "{} reports a failed capture write — that pass stopped recording mid-run, so its ledger is \
             silently short and no report can be derived from it",
            broken.join(", ")
        ));
    }
    let build = build_clips(&capture)?;
    let halves = half_names(&capture);
    let Some(selection) = halves.iter().position(|half| half == SELECTION_HALF) else {
        return Err(anyhow!(
            "{FILE_PARTITION} carries no {SELECTION_HALF} half — the capture does not match the frozen \
             selection half"
        ));
    };
    let Some(held_out) = halves.iter().position(|half| half != SELECTION_HALF) else {
        return Err(anyhow!(
            "{FILE_PARTITION} carries no half other than {SELECTION_HALF} — the frozen partition needs a \
             held-out half"
        ));
    };

    // ── Step 1: every readout under the streaming reduction ──
    let mut candidates = step_one_candidates();
    let mut evals: Vec<Vec<HalfEval>> = candidates
        .iter()
        .map(|candidate| {
            let candidate_triggers = triggers_for_candidate(*candidate, &build.clips);
            evaluate_candidate_halves(&halves, selection, &build.clips, &candidate_triggers)
        })
        .collect();

    // ── Step 2: select on the selection half, from exactly the step-1 list ──
    let step_one = candidates.len();
    let mut ranked: Vec<usize> = (0..step_one).collect();
    ranked.sort_by(|&a, &b| selection_cmp(&evals[a][selection], &evals[b][selection], a, b));
    let winner = ranked[0];
    let runner_up = ranked.get(1).copied();
    let auc_best = (0..step_one).min_by(|&a, &b| {
        evals[a][selection]
            .auc
            .partial_cmp(&evals[b][selection].auc)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    let auc_best = auc_best.expect("the candidate list is never empty");

    // ── Step 3: the three clip-level reductions for exactly two readouts ──
    for candidate in step_three_candidates(candidates[winner].readout) {
        let candidate_triggers = triggers_for_candidate(candidate, &build.clips);
        let candidate_evals =
            evaluate_candidate_halves(&halves, selection, &build.clips, &candidate_triggers);
        candidates.push(candidate);
        evals.push(candidate_evals);
    }

    let control = candidates
        .iter()
        .position(|candidate| *candidate == CONTROL)
        .expect("the control is always the first candidate");

    // ── Report pieces ──
    let (families, half_groups) = group_counts(&build, &halves);
    let mut integrity = Integrity {
        clips: build.clips.len(),
        enrollment: EnrollmentCounts {
            utterances: build.enrollment_matrices.len(),
            tokens: build.enrollment_tokens.len(),
            dim: build
                .enrollment_matrices
                .first()
                .map_or(0, |(_, _, dim)| *dim),
        },
        skipped_windows: build.skipped_windows.clone(),
        skipped_enrollment: build.skipped_enrollment.clone(),
        windows_without_partition: build.windows_without_partition.clone(),
        partial_tail_records: capture.partial_tails.clone(),
        blob_bytes: build.blob_bytes,
        referenced_bytes: build.referenced_bytes,
        missing_clip_results: PASSES
            .iter()
            .map(|pass| (pass.to_string(), missing_clip_results(&capture, pass)))
            .collect(),
        run_linkage: run_linkage(&capture)?,
        manifest_mismatches: manifest_mismatches(&capture),
        unpaired_windows: PASSES
            .iter()
            .filter_map(|pass| {
                capture.metas[*pass]
                    .unpaired_windows
                    .map(|count| ((*pass).to_string(), count))
            })
            .collect(),
        warnings: Vec::new(),
        families,
        halves: half_groups,
        corpus: CorpusCounts {
            baseline: capture.metas[PASS_BASELINE].clone(),
            measure: capture.metas[PASS_MEASURE].clone(),
        },
    };
    integrity.warnings = integrity_warnings(&integrity);

    let per_clip = build
        .clips
        .iter()
        .map(|clip| ClipReport {
            label: clip.label.clone(),
            family: clip.family.clone(),
            half: clip.half.clone(),
            non_flush_windows: clip.non_flush_windows(),
            flush_windows: clip.flush_windows,
        })
        .collect();

    let id_of = |index: usize| CandidateId {
        readout: candidates[index].readout.name(),
        reduction: candidates[index].reduction.name(),
        summary_selection: evals[index][selection].summary_statistic,
        summary_held_out: evals[index][held_out].summary_statistic,
        auc_selection: evals[index][selection].auc,
    };
    let tie_break = runner_up.and_then(|runner| {
        (evals[winner][selection].summary_statistic == evals[runner][selection].summary_statistic)
            .then(|| {
                tie_break_reason(&evals[winner][selection], &evals[runner][selection]).to_string()
            })
    });
    let selection_report = SelectionReport {
        selection_half: SELECTION_HALF.to_string(),
        rule: SELECTION_RULE.to_string(),
        winner: id_of(winner),
        runner_up: runner_up.map(id_of),
        auc_best: id_of(auc_best),
        auc_best_disagrees: auc_best != winner,
        tie_break,
    };

    // ── Verdicts against the control ──
    let baseline = baseline_report(&capture);
    let spread_held_out = baseline
        .difference
        .negative_spread
        .get(&halves[held_out])
        .copied()
        .unwrap_or(0);
    let verdicts: Vec<Verdict> = (0..candidates.len())
        .filter(|&index| index != control)
        .map(|index| {
            let candidate_half = &evals[index][held_out];
            let control_half = &evals[control][held_out];
            let movement =
                as_i64(control_half.summary_statistic) - as_i64(candidate_half.summary_statistic);
            let selection_advantage = evals[control][selection].summary_statistic
                > evals[index][selection].summary_statistic;
            let lift_held_out: Vec<LiftPoint> = candidate_half
                .fixed_levels
                .iter()
                .zip(&control_half.fixed_levels)
                .map(|(candidate, control)| LiftPoint {
                    target_recall: control.target_recall,
                    control_level: control.level,
                    candidate_level: candidate.level,
                    lift: as_i64(control.firing_negatives) - as_i64(candidate.firing_negatives),
                })
                .collect();
            let lifted = lift_held_out
                .iter()
                .all(|point| point.lift > as_i64(spread_held_out));
            Verdict {
                readout: candidates[index].readout.name(),
                reduction: candidates[index].reduction.name(),
                control_summary_selection: evals[control][selection].summary_statistic,
                candidate_summary_selection: evals[index][selection].summary_statistic,
                control_summary_held_out: control_half.summary_statistic,
                candidate_summary_held_out: candidate_half.summary_statistic,
                movement,
                spread_held_out,
                lift_held_out,
                lifted,
                rule: verdict_rule(movement, selection_advantage, lifted),
            }
        })
        .collect();

    // Consumed last: the report owns the evaluations, the verdicts only read
    // them.
    let candidate_reports: Vec<CandidateReport> = candidates
        .iter()
        .zip(evals)
        .enumerate()
        .map(|(index, (candidate, halves))| CandidateReport {
            readout: candidate.readout.name(),
            reduction: candidate.reduction.name(),
            enumeration_order: index,
            halves,
        })
        .collect();

    Ok(Report {
        capture_dir: dir.display().to_string(),
        files_read: capture.files.clone(),
        timestamp: crate::db::now(),
        integrity,
        comparison_space: ComparisonSpace {
            similarity: "crate::vector::cosine_similarity (the product's helper; normalized vectors, clamped to \
                         [0, 1])",
            min_window_value: build.min_window_value,
            note: "the frozen sweep still spans −1.0 … 1.0, but its [-1, 0) half fires nothing because no \
                   compared value can be negative — no floor, threshold or veto is involved",
        },
        partition: partition_report(&capture, &halves),
        baseline,
        candidates: candidate_reports,
        per_clip,
        selection: selection_report,
        verdicts,
        adoptability: adoptability(&candidates),
        caveats: caveats(),
    })
}

/// The step-2 procedure, stated in the report.
const SELECTION_RULE: &str = "the winner is the candidate with the lowest summary statistic (the sum of the six \
     fixed-level firing-negative counts) on the selection half, chosen among the eight readouts under the \
     streaming reduction (`sliding3`) — the clip-level reductions do not exist at that point; ties break on the \
     lower firing-negative count at the highest attainable fixed recall level, then on the lower AUC, then on the \
     fixed enumeration order (mean, component_max, last_k_1, last_k_2, last_k_3, magnitude_weighted_mean, \
     token_best_max, token_best_mean).  Only afterwards are the three clip-level reductions computed, for the \
     reference readout and the winner, taking no part in the selection.";

/// `n` as `i64` — counts and lifts fit comfortably.
#[expect(
    clippy::cast_possible_wrap,
    reason = "analysis counts are tiny — no sign change"
)]
#[must_use]
fn as_i64(n: usize) -> i64 {
    n as i64
}

// ── Rendering ────────────────────────────────────────────────────────────

/// Print the run log's digest of the report: where the numbers live, how much
/// was analysed, the selection winner and the readings that make the numbers
/// interpretable.  Every number itself lives only in `analysis.json`.
fn print_summary(report: &Report) {
    let mut out = std::io::stdout();
    let winner = &report.selection.winner;
    let _ = writeln!(
        out,
        "wake analysis — wrote {}, {} clips, {} candidates, {} verdicts, selection winner {} / {}",
        Path::new(&report.capture_dir).join(ANALYSIS_JSON).display(),
        report.integrity.clips,
        report.candidates.len(),
        report.verdicts.len(),
        winner.readout,
        winner.reduction
    );
    let _ = writeln!(out, "  readings: {}", readings(report));
    for warning in &report.integrity.warnings {
        let _ = writeln!(out, "warning: {warning}");
    }
}

/// The conventions that fix every reported number, printed with the digest
/// because they are not legible from the curves alone.
fn readings(report: &Report) -> String {
    // The held-out half's differing-negative count is the tolerance rule two
    // uses; a half the map does not name has no differing negative at all.
    let spread = report
        .partition
        .halves
        .iter()
        .find(|half| half.role != "selection")
        .and_then(|half| report.baseline.difference.negative_spread.get(&half.half))
        .copied()
        .unwrap_or(0);
    format!(
        "entry (3) `last_k` competes in the selection as three curves (k = 1, 2, 3) and its k = 1 curve is \
         the entry's representative in the extra frame reductions; entry (6) weights the raw (un-normalised) \
         token norms; a fixed recall level is mapped to the tightest swept level that attains it; rule two's \
         tolerance is the observed held-out negative spread ({spread}), not the pass-to-pass positive \
         difference"
    )
}

// ── Entry point ──────────────────────────────────────────────────────────

/// The report file the analyser writes into the capture directory.
const ANALYSIS_JSON: &str = "analysis.json";

/// Run the offline analyser over the capture directory.
///
/// Resolves the capture directory through
/// [`capture_dir`](crate::audio::wake_capture::capture_dir) — setting the
/// storage root first when the bench's earlier setup did not, because the
/// analyser may run standalone — then reads every capture file, writes
/// `analysis.json` into the directory and prints its digest to stdout.
///
/// Missing, unreadable or inconsistent capture files end the process with
/// status 1 and a message on stderr: the analyser never guesses around a broken
/// capture.  A partial trailing record is tolerated — the writer appends
/// unbuffered, so a killed process leaves one — and reported in
/// `partial_tail_records`.
pub(crate) fn run() {
    if crate::config::CONFIG.try_storage_root().is_none() {
        match crate::config::default_config_dir() {
            Ok(root) => crate::config::CONFIG.set_storage_root(root),
            Err(error) => fail(&anyhow!("cannot resolve the storage root: {error:#}")),
        }
    }
    let dir: PathBuf = capture_dir();
    let report = match analyse(&dir) {
        Ok(report) => report,
        Err(error) => fail(&error),
    };
    let json = match serde_json::to_string_pretty(&report) {
        Ok(json) => json,
        Err(error) => fail(&anyhow!("cannot serialize {ANALYSIS_JSON}: {error}")),
    };
    if let Err(error) = fs::write(dir.join(ANALYSIS_JSON), &json) {
        fail(&anyhow!(
            "cannot write {ANALYSIS_JSON} in {}: {error}",
            dir.display()
        ));
    }
    print_summary(&report);
}

/// Report a fatal analysis error and exit with status 1.
fn fail(error: &anyhow::Error) -> ! {
    eprintln!("wake analysis: {error:#}");
    std::process::exit(1);
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert a float slice against a hand-computed expectation.
    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "length {actual:?}");
        for (value, expected) in actual.iter().zip(expected) {
            assert!(
                (value - expected).abs() < 1e-5,
                "expected {expected:?}, got {actual:?}"
            );
        }
    }

    // ── Reading ──────────────────────────────────────────────────────────

    /// A minimal JSONL record for the [`parse_jsonl`] rule.
    #[derive(Debug, Deserialize)]
    struct Tiny {
        n: u64,
    }

    #[test]
    fn parse_jsonl_skips_only_a_partial_trailing_record() {
        // The writer appends unbuffered, so a killed process leaves a cut-off
        // last line: that one is skipped and recorded.
        let mut partial_tails = Vec::new();
        let records: Vec<Tiny> = parse_jsonl(
            "t.jsonl",
            "{\"n\": 1}\n{\"n\": 2}\n{\"n\":",
            &mut partial_tails,
        )
        .unwrap();
        let values: Vec<u64> = records.iter().map(|record| record.n).collect();
        assert_eq!(values, [1, 2]);
        assert_eq!(partial_tails.len(), 1);
        assert_eq!(
            partial_tails[0],
            "t.jsonl: line 3 is a partial trailing record and was skipped"
        );

        // The same cut-off line before a good one cannot be a tail: hard error.
        let mut partial_tails = Vec::new();
        let error = parse_jsonl::<Tiny>(
            "t.jsonl",
            "{\"n\": 1}\n{\"n\":\n{\"n\": 3}\n",
            &mut partial_tails,
        )
        .expect_err("a partial record before the end is not a tail");
        assert!(error.to_string().contains("line 2"), "{error}");
        assert!(partial_tails.is_empty());
    }

    // ── Sweep mapping ────────────────────────────────────────────────────

    #[test]
    fn tightest_level_picks_the_highest_matching_level() {
        // Levels ascend with the index; recall 1.0 up to index 3, then 0.5, then
        // 0.0 — a plateau in both directions.
        let recall = [1.0, 1.0, 1.0, 0.5, 0.5, 0.0];
        assert_eq!(tightest_level(&recall, 0.5), 4, "plateau maps to its top");
        assert_eq!(tightest_level(&recall, 1.0), 2, "plateau maps to its top");
        assert_eq!(
            tightest_level(&recall, 0.0),
            5,
            "the loosest target still lands on the highest level attaining it"
        );
    }

    #[test]
    fn tightest_level_falls_back_to_the_most_permissive_level() {
        // The selection half only reaches recall 0.5, so R = 1.00 is
        // unattainable: the nearest attainable point is L = −1.0 (index 0).
        let recall = [0.5, 0.5, 0.0];
        assert_eq!(tightest_level(&recall, 1.0), 0);
        assert_eq!(tightest_level(&recall, 0.9), 0);
        assert_eq!(tightest_level(&recall, 0.5), 1);
    }

    #[test]
    fn level_index_geometry_is_the_fixed_sweep() {
        assert_eq!(LEVEL_STEPS, 2001);
        assert!((level_of(0) - (-1.0)).abs() < 1e-6);
        assert!(level_of(LEVEL_ZERO_INDEX).abs() < 1e-6);
        assert!((level_of(LEVEL_STEPS - 1) - 1.0).abs() < 1e-6);
        assert!((level_of(LEVEL_ZERO_INDEX + 1) - 0.001).abs() < 1e-6);
    }

    // ── Frame reductions ─────────────────────────────────────────────────

    #[test]
    fn sliding3_is_the_sliding_mean_of_three() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_close(&sliding3(&values), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn sliding3_degrades_to_the_mean_below_three_windows() {
        assert_close(&sliding3(&[2.0, 4.0]), &[3.0]);
        assert_close(&sliding3(&[7.0]), &[7.0]);
        assert!(sliding3(&[]).is_empty(), "no windows means no trigger");
    }

    #[test]
    fn clip_reductions_collapse_to_one_trigger_and_none_when_empty() {
        let values = [1.0, 4.0, 3.0];
        assert_close(&triggers(Reduction::ClipMax, &values), &[4.0]);
        assert_close(&triggers(Reduction::ClipMean, &values), &[8.0 / 3.0]);
        assert_close(&triggers(Reduction::ClipMedian, &values), &[3.0]);
        assert!(triggers(Reduction::ClipMax, &[]).is_empty());
        assert!(triggers(Reduction::ClipMedian, &[]).is_empty());
    }

    #[test]
    fn zero_window_clip_fires_at_no_level() {
        let clip = ClipData {
            family: POSITIVE_FAMILY.to_string(),
            label: "p".to_string(),
            half: "A".to_string(),
            values: Vec::new(),
            flush_windows: 2,
        };
        let triggers = triggers_for_candidate(CONTROL, std::slice::from_ref(&clip));
        assert!(triggers[0].is_empty());
        let sweep = sweep_half("A", std::slice::from_ref(&clip), &triggers);
        let applied = own_level_indices(&sweep);
        let eval = evaluate_sweep(sweep, &applied);
        assert_eq!(eval.positives, 1, "the clip stays in the denominator");
        assert_eq!(eval.positives_without_windows, 1);
        // A recall of zero at the loosest level: never a fire.
        assert!(
            eval.fixed_levels
                .iter()
                .all(|level| level.attained_recall.abs() < 1e-6)
        );
    }

    // ── Readouts ─────────────────────────────────────────────────────────

    #[test]
    fn vector_readouts_match_hand_computed_values() {
        // Two tokens of dim 2: [[1, 2], [3, 4]].  `component_max` and `last_k_*`
        // reduce L2-normalised tokens — t0 = [1, 2]/√5, t1 = [3, 4]/5 =
        // [0.6, 0.8] — while `mean` (and the weighted mean, tested separately)
        // keeps the raw rows.
        let matrix = [1.0f32, 2.0, 3.0, 4.0];
        let sqrt5 = 5.0f32.sqrt();
        assert_close(
            &readout_vector(Readout::Mean, &matrix, 2, 2).unwrap(),
            &[2.0, 3.0],
        );
        assert_close(
            &readout_vector(Readout::ComponentMax, &matrix, 2, 2).unwrap(),
            &[0.6, 2.0 / sqrt5],
        );
        assert_close(
            &readout_vector(Readout::LastK(1), &matrix, 2, 2).unwrap(),
            &[0.6, 0.8],
        );
        assert_close(
            &readout_vector(Readout::LastK(2), &matrix, 2, 2).unwrap(),
            &[
                f32::midpoint(0.6, 1.0 / sqrt5),
                f32::midpoint(0.8, 2.0 / sqrt5),
            ],
        );
        assert!(readout_vector(Readout::TokenBestMax, &matrix, 2, 2).is_none());
    }

    #[test]
    fn magnitude_weighted_mean_weights_before_normalization() {
        // Tokens [1, 0] (norm 1) and [0, 4] (norm 4).  Weighted by the raw
        // norms the mean is (1·[1,0] + 4·[0,4]) / 5 = [0.2, 3.2]; weighting
        // after normalization would give [0.5, 2.0] instead.  L2-normalization
        // is the caller's step, so the raw weighted mean is what is asserted.
        let matrix = [1.0f32, 0.0, 0.0, 4.0];
        let weighted = readout_vector(Readout::MagnitudeWeightedMean, &matrix, 2, 2).unwrap();
        assert_close(&weighted, &[0.2, 3.2]);
        assert!(weighted[1] > weighted[0], "the loud token dominates");
    }

    #[test]
    fn token_best_readouts_match_hand_computed_cosines() {
        // Enrolment pool: the two unit axes.  Window tokens: [1, 1] (cos 1/√2
        // to either axis) and [3, 0] (cos 1.0 to the first axis).
        let pool = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];
        let matrix = [1.0f32, 1.0, 3.0, 0.0];
        let bests = token_best_cosines(&matrix, 2, 2, &pool);
        let diagonal = 1.0 / 2.0f32.sqrt();
        assert!((bests[0] - diagonal).abs() < 1e-5, "{bests:?}");
        assert!((bests[1] - 1.0).abs() < 1e-5, "{bests:?}");

        // `build_clips` refuses an enrolment-less capture, so every vector
        // readout has a prototype here; the scalar slots ignore theirs.
        let prototypes: [Option<Vec<f32>>; Readout::COUNT] =
            std::array::from_fn(|_| Some(vec![1.0f32, 0.0]));
        let values = window_values(&matrix, 2, 2, &prototypes, &pool);
        assert!((values[Readout::TokenBestMax.index()] - 1.0).abs() < 1e-5);
        assert!(
            (values[Readout::TokenBestMean.index()] - f32::midpoint(1.0, diagonal)).abs() < 1e-5,
            "{values:?}"
        );
    }

    #[test]
    fn prototype_averages_per_utterance_readouts_not_similarities() {
        // Two utterances with orthogonal means: the averaged prototype has both
        // components, so a window equal to either utterance scores 1/√2 — NOT
        // 1.0 as a mean of per-utterance similarities would.
        let utterances = vec![
            (vec![1.0f32, 0.0], 1usize, 2usize),
            (vec![0.0f32, 1.0], 1usize, 2usize),
        ];
        let proto = prototype(Readout::Mean, &utterances).unwrap();
        assert_close(&proto, &[1.0 / 2.0f32.sqrt(), 1.0 / 2.0f32.sqrt()]);
        let window = readout_vector(Readout::Mean, &[1.0f32, 0.0], 1, 2).unwrap();
        let mut normalized = window;
        l2_normalize_in_place(&mut normalized);
        assert!((cosine_similarity(&normalized, &proto) - 1.0 / 2.0f32.sqrt()).abs() < 1e-5);
    }

    // ── Verdicts ─────────────────────────────────────────────────────────

    #[test]
    fn verdict_precedence_follows_the_rule_order() {
        // Rule one wins even when the movement would otherwise be rule three's.
        assert!(verdict_rule(2, true, false).starts_with("rule one"));
        assert!(verdict_rule(-2, true, false).starts_with("rule one"));
        // `lifted` implies a strictly better movement, so it is checked before
        // the held-out-loss rules.
        assert_eq!(verdict_rule(7, false, true), "separation lifted");
        // Rule two is the same exchange with the separation not lifted.
        assert!(verdict_rule(7, true, false).starts_with("rule two"));
        // Rule three needs a selection-half advantage that reverses.
        assert!(verdict_rule(-7, true, false).starts_with("rule three"));
        // Rule four is a held-out loss without the selection-half advantage.
        assert!(verdict_rule(-7, false, false).starts_with("rule four"));
    }

    // ── Candidate list ───────────────────────────────────────────────────

    #[test]
    fn step_three_extras_are_the_clip_reductions_of_two_readouts() {
        // The step-1 list plus these extras is the whole evaluated set: 8
        // sliding3 readouts + 3 clip reductions for `mean` + 3 for the last_k_1
        // representative.
        let extras = step_three_candidates(Readout::LastK(2));
        assert_eq!(extras.len(), 6);
        assert!(
            extras
                .iter()
                .all(|candidate| candidate.reduction != Reduction::Sliding3),
            "every extra is a clip-level reduction"
        );
        let mut unique = extras.clone();
        unique.sort_by_key(|candidate| (candidate.readout.name(), candidate.reduction.name()));
        unique.dedup();
        assert_eq!(unique.len(), extras.len());

        // When the winner is `mean` itself the two readouts coincide, so the
        // extras collapse to three.
        let extras = step_three_candidates(Readout::Mean);
        assert_eq!(extras.len(), 3);
        assert!(
            extras
                .iter()
                .all(|candidate| candidate.readout == Readout::Mean),
            "both extra readouts are `mean`"
        );
    }

    // ── Half evaluation ──────────────────────────────────────────────────

    /// A one-window clip whose every readout slot carries `value`, so
    /// `sliding3` reduces it to that value.
    fn clip(family: &str, label: &str, half: &str, value: f32) -> ClipData {
        ClipData {
            family: family.to_string(),
            label: label.to_string(),
            half: half.to_string(),
            values: vec![[value; Readout::COUNT]],
            flush_windows: 0,
        }
    }

    #[test]
    fn evaluate_sweep_counts_recall_and_false_accepts() {
        // One positive window value and one negative window value, both above
        // the mid sweep: at any level ≤ 0.5 both fire.
        let clips = vec![
            clip(POSITIVE_FAMILY, "p", "A", 0.5),
            clip("noise", "n", "A", 0.5),
        ];
        let triggers = triggers_for_candidate(CONTROL, &clips);
        let sweep = sweep_half("A", &clips, &triggers);
        let applied = own_level_indices(&sweep);
        let eval = evaluate_sweep(sweep, &applied);
        assert_eq!(eval.positives, 1);
        assert_eq!(eval.negatives, 1);
        assert_eq!(eval.positives_without_windows, 0);
        // The fixed levels are descending targets: recall 1.00 is attainable
        // (the single positive always fires at or below its own value).
        assert!((eval.fixed_levels[0].attained_recall - 1.0).abs() < 1e-6);
        assert_eq!(eval.fixed_levels[0].firing_negatives, 1);
        assert_eq!(
            eval.summary_statistic, 6,
            "the negative fires at every level"
        );
        // `sliding3` of a single window is that window's value.
        assert_close(&triggers[0], &[0.5]);
    }

    #[test]
    fn held_out_cells_use_the_selection_halfs_levels() {
        // Half A's positives [1.0, 0.5] map every target to level 0.5 except
        // R = 0.50, which maps to 1.0.  Half B's positives [0.8, 0.0] attain
        // far less at those applied levels, so B's authoritative cells must
        // carry the applied level and B's own — lower — attained recall.
        let clips = vec![
            clip(POSITIVE_FAMILY, "pa1", "A", 1.0),
            clip(POSITIVE_FAMILY, "pa2", "A", 0.5),
            clip("noise", "na", "A", 0.6),
            clip(POSITIVE_FAMILY, "pb1", "B", 0.8),
            clip(POSITIVE_FAMILY, "pb2", "B", 0.0),
            clip("noise", "nb", "B", 0.4),
        ];
        let triggers = triggers_for_candidate(CONTROL, &clips);
        let halves = vec!["A".to_string(), "B".to_string()];
        let evals = evaluate_candidate_halves(&halves, 0, &clips, &triggers);
        let (selection, held_out) = (&evals[0], &evals[1]);

        // The selection half maps itself, so it attains every target.
        for level in &selection.fixed_levels {
            assert!(
                level.attained_recall >= level.target_recall,
                "target {} attained {:.3}",
                level.target_recall,
                level.attained_recall
            );
        }
        let selection_top = &selection.fixed_levels[0];
        assert!(
            (selection_top.level - 0.5).abs() < 1e-6,
            "selection level {}",
            selection_top.level
        );
        assert!((selection_top.attained_recall - 1.0).abs() < 1e-6);

        // Held-out R = 1.00: the applied level is A's 0.5, where B only
        // attains 0.5 — the shortfall is reported, not corrected.
        let held_top = &held_out.fixed_levels[0];
        assert!(
            (held_top.level - 0.5).abs() < 1e-6,
            "applied, not re-derived"
        );
        assert!(
            (held_top.attained_recall - 0.5).abs() < 1e-6,
            "held-out attained {}",
            held_top.attained_recall
        );
        assert!(held_top.attained_recall < held_top.target_recall);

        // Held-out R = 0.50: the applied level is A's own 1.0, where B attains
        // nothing at all.
        let held_bottom = &held_out.fixed_levels[5];
        assert!((held_bottom.level - 1.0).abs() < 1e-6);
        assert!(held_bottom.attained_recall.abs() < 1e-6);
    }
}
