//! Verifier for wake word false-trigger suppression.
//!
//! Implements a lightweight second-stage classifier that runs AFTER the
//! Conv1D classifier fires, as an additional AND gate.
//!
//! ## Architecture: Conv1D
//!
//! Conv1D(96→2, k=3, padding=1) → LeakyReLU → AdaptiveAvgPool → Linear(2→1) → Sigmoid.
//! ~581 trainable parameters. Preserves temporal structure across the 3-frame
//! window (288-dim concatenated input, no mean-pooling).
//!
//! LeakyReLU (mahbot-1008 Fix 3) replaces ReLU so the feature path cannot
//! collapse to a dead zone: with ReLU, any input whose Conv1D pre-activations
//! are all ≤ 0 produced pooled features `[0, 0]` and a constant
//! `sigmoid(fc_bias)` output regardless of input (the observed `6.67e-8`
//! reject-all floor).
//!
//! When not trained, the verifier acts as a no-op (all frames pass).
//!
//! # Architecture
//!
//! Training pipeline: per-frame embeddings → windowing (concatenated 288-dim)
//! → L2-norm → Adam training. Inference is ~3μs per frame.
//!
//! ## Training data
//!
//! - **Positive examples**: 3-frame stride-1 windows formed from enrollment
//!   utterance per-frame embeddings.
//! - **Negative examples**: Distribution-matched synthetic negatives from
//!   positive statistics (bootstrapping) or hard-negative embeddings collected
//!   from near-miss frames during detection.
//! - **Confusable negatives**: Pre-computed near-miss phrase embeddings (e.g.
//!   "hey map bot", "day mahbot") with 15× higher per-example weight during
//!   training so the verifier learns to reject confusable phrases.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::audio::embedding_sequence::EmbeddingSequence;
use crate::audio::embedding_sequence::Source;
use crate::{EMBEDDING_DIM, VERIFIER_INPUT_DIM, VERIFIER_WINDOW_SIZE};

/// Default decision threshold for the verifier.
///
/// **Since mahbot-997:** This constant now serves as the **fallback threshold**
/// used when auto-calibration is skipped (e.g. training with only synthetic
/// negatives, or insufficient validation data).  When the verifier is trained
/// with real (non-synthetic) negative sequences, [`train`](VoiceVerifier::train)
/// automatically calibrates a data-driven threshold via
/// [`calibrate_verifier_threshold`] — see Constrained Weighted Youden's J
/// (mahbot-997) for details.
///
/// The pre-mahbot-997 calibration is preserved for reference:
/// Calibrated for **dense stride-8 embeddings** (mahbot-923).  The original
/// threshold of 0.60 was calibrated against streaming embeddings (mahbot-890).
/// After the pipeline-wide switch to dense stride-8 (mahbot-923), all trainable
/// components — classifier and verifier — produce scores with a different
/// scale.  A uniform 1.58× multiplier was derived empirically from the
/// distribution shift between streaming and dense embeddings:
///
/// ```text
/// new_threshold = old_threshold × 1.58
/// 0.948 ≈ 0.60 × 1.58
/// ```
///
/// The 1.58× factor was validated by comparing score distributions of the
/// production classifier and verifier across a held-out set of enrollment
/// utterances after the dense-only migration.  The old sweep results
/// (mahbot-890, 0.05 increments from 0.40 to 0.70) are not directly
/// transferable since they were measured against the streaming pipeline
/// distribution, but the multiplier preserves the calibration relationship.
///
/// ## Sweep reference (original, mahbot-890, streaming distribution)
///
/// | Threshold | Detection rate (range) | Mean DR | Verifier-pass FA / run | HARD pass rate |
/// |-----------|----------------------|---------|----------------------|----------------------------------|
/// | 0.40      | 92.3%                | 92.3%   | 4                     | ✗ (conf=2, total=4) |
/// | 0.45      | 84.6%                | 84.6%   | 3                     | ✗ (conf=1, total=3) |
/// | 0.50      | 53.8%                | 53.8%   | 2                     | ✗ (conf=1, total=2) |
/// | 0.55      | 84.6–92.3%           | 89.2%   | 1.75                  | 3/5 (60%) |
/// | **0.60**  | **76.9–92.3%**       | **87.7%** | **1.0**              | **4/5 (80%)** |
/// | 0.65      | 84.6%                | 84.6%   | 1.0                   | 2/3 (67%) |
/// | 0.70      | 84.6%                | 84.6%   | 2                     | ✗ (conf=2, total=3) |
///
/// **Previously:** 0.60 (streaming, mahbot-890), 0.50 (mahbot-882),
/// 0.4 (mahbot-853), 0.6 (mahbot-829), 0.5 (mahbot-797), 0.3 (mahbot-788).
pub(crate) const DEFAULT_VERIFIER_THRESHOLD: f32 = 0.948;

/// **Constant** verifier acceptance floor for the wake-word detection
/// confirmation gate (mahbot-1023).
///
/// The user-approved production semantics are: *"the verifier accepts the
/// enrolled voiceprint at 0.86"*.  Every detection confirmation gate
/// (`score_single_embedding` candidate confirmation, the no-candidate
/// fallback gate, `is_collapsed`) uses this **constant** — NOT the
/// entropy-seeded runtime-calibrated `VoiceVerifier::threshold` (which
/// drifted 0.86 → 0.91 across two identical-code runs) and NOT the
/// [`DEFAULT_VERIFIER_THRESHOLD`] fallback (0.948).  The runtime-calibrated
/// value and the fallback remain report-only references so threshold drift is
/// observable without affecting product behavior.
///
/// FP-safety is measured: the highest negative verifier peak on the
/// approved 59-negative corpus is 0.7587 (`day mahbot_s2`, the mahbot-1022
/// pre-deferral baseline); the archive-worst per-frame negative verifier
/// peak is **0.8396** (run 20260801-085346, margin 0.0204 below the floor),
/// NOT 0.6833 — the 0.6833 reading (run 20260801-061348) is the archive
/// second-worst and was previously misquoted as the worst.  The fresh-run
/// (final-code) worst per-frame negative verifier max is **0.7295** (run
/// 20260801-090824, margin 0.1305 below the floor); the fresh-run
/// margin-to-floor distribution is 0.7901 / 0.2680 / 0.1305 across runs
/// 090605 / 090719 / 090824.  0 false accepts in every archived run
/// (0/177 total on the 59-negative corpus).  The binding positive variant
/// is speed_down: there are **two** observed sub-floor peaks — 0.7886
/// (run 20260801-061348) and 0.8090 (run 20260801-085648) — not one; the
/// fresh-run speed_down peaks are 0.9485 / 0.8983 / 0.8620 across runs
/// 090605 / 090719 / 090824 (the 0.8620 reading cleared the floor by
/// 0.002).  Do NOT quote noise_overlap verifier peaks as negative-FA
/// margins: the noise_overlap cells are positive wake-word utterances
/// (cross-speaker TTS audio), not negative-corpus evidence (mahbot-1025).
/// The floor stays at 0.86; any future re-derivation must stay above a
/// freshly-measured negative peak and is out of scope for this ticket
/// (mahbot-1024 re-scope: the strict ring-4 formula is report-only
/// diagnostics; acceptance is mean TP ≥ 4/5 across ≥ 3 fresh runs).
pub(crate) const VERIFIER_ACCEPTANCE_FLOOR: f32 = 0.86;

/// Fraction of sequences held out for validation (80/20 train/val split).
///
/// Split is per-sequence (avoiding data leakage from overlapping windows) with
/// stratification preserving both pos:neg ratio AND negative tier proportions
/// (confusable/unrelated/ambient/owner).  Positive sequences prefer the
/// provenance-group holdout (mahbot-1008 Fix 1); when no group can be held
/// out the per-sequence split applies.  If the split would leave fewer than
/// [`MIN_POSITIVE_WINDOWS`] positive windows in training — or produce no
/// validation data at all — training runs on ALL sequences with empty
/// validation (no leaky per-window fallback, mahbot-1008/mahbot-1011).
pub(crate) const VALIDATION_SPLIT: f32 = 0.2;

/// Log training and validation loss every N iterations.
const LOG_LOSS_INTERVAL: usize = 50;

// ── Conv1D verifier training hyperparameters (mahbot-995) ────────────────
//
// Matches the Conv1D classifier's proven values (wake_word_classifier.rs)
// for the same small-dataset (<200 positive windows) regime.

/// Learning rate for Conv1D verifier Adam training.
const CONV_LEARNING_RATE: f32 = 0.001;
/// L2 regularization strength for Conv1D verifier.
pub(crate) const CONV_L2_LAMBDA: f32 = 0.0001;
/// Batch size for Conv1D verifier mini-batch training.
const CONV_BATCH_SIZE: usize = 32;
/// Maximum training epochs for Conv1D verifier.
const CONV_MAX_EPOCHS: usize = 100;
/// Early stopping patience for Conv1D verifier.
const CONV_EARLY_STOP_PATIENCE: usize = 15;
/// Conv1D output channels for verifier (96→CONV_VERIFIER_OUT).
pub(crate) const CONV_VERIFIER_OUT: usize = 2;
/// Conv1D kernel size for verifier.
pub(crate) const CONV_VERIFIER_KERNEL_SIZE: usize = 3;

/// Minimum number of positive **windows** required to train the verifier
/// (mahbot-1008 Fix 2).
///
/// The pre-fix guard counted positive *sequences* (≥5 utterances with ≥3
/// embeddings each) — but an utterance with 3 embeddings produces exactly 1
/// window, so the verifier could train on as few as 5 positive windows and
/// memorize them into a reject-all brick wall.  Training with fewer than this
/// many positive windows returns an **untrained no-op** (all frames pass)
/// instead of a trained reject-all.
///
/// The check lives in [`prepare_training_data`] so it covers every call site
/// (production, the synthetic-negatives fallback, and the E2E benchmark) and
/// also catches the stricter failure mode where positive sequences exist but
/// produce **zero** windows (all utterances < [`VERIFIER_WINDOW_SIZE`] frames)
/// — which previously trained an all-negative reject-all with `trained: true`.
pub(crate) const MIN_POSITIVE_WINDOWS: usize = 30;

/// Cap on the per-positive-window class weight (mahbot-1008 Fix 4).
///
/// The observed failure trained on 58 positive windows against 11,074
/// negatives with per-example positive weight ≈ 2,208× — the model memorized
/// the 58 positives instead of learning a generalizable policy.  Capping the
/// weight prevents per-example memorization while keeping a strong positive
/// signal.  Negative subsampling is deliberately NOT performed: the full
/// negative set (especially confusable/unrelated tiers) is what suppresses
/// false accepts, and the tiered FA limits in the E2E benchmark were tuned
/// against it.
pub(crate) const MAX_CLASS_WEIGHT: f32 = 50.0;

/// Slope of the verifier-local LeakyReLU activation (mahbot-1008 Fix 3).
///
/// ReLU's dead zone was the root cause of the constant-floor collapse: when
/// every Conv1D pre-activation is ≤ 0 the pooled features are `[0, 0]` and the
/// logit degenerates to `fc_bias` — an input-independent constant (observed
/// `6.67e-8`).  LeakyReLU keeps a small gradient (and a small, input-dependent
/// signal) in the negative half-plane so the feature path can never die
/// completely.
///
/// Deliberately verifier-local: the classifier's `relu()` helper stays
/// unchanged (the classifier has its own calibration and is out of scope for
/// mahbot-1008).
const LEAKY_RELU_SLOPE: f32 = 0.01;

/// Hard clamp on the verifier's `fc_bias` (mahbot-1008 Fix 3).
///
/// Unregularized and unpulled-up by any positive signal, `fc_bias` drifted to
/// −16.52 (sigmoid ≈ 6.67e-8) under ~85% negative-only batches.  Bounding it
/// to ±3 keeps the input-independent component of the logit inside
/// `sigmoid(±3) ∈ [0.047, 0.953]` so a dead feature path can never produce a
/// lethal sub-`1e-6` reject floor.  Combined with LeakyReLU the logit is still
/// input-dependent; the clamp only bounds the constant component.
const FC_BIAS_CLAMP: f32 = 3.0;

/// Minimum held-out (out-of-session) validation TPR for a trained verifier.
///
/// When honest validation data is available and the trained verifier's TPR at
/// the selected threshold falls below this, training emits a prominent warning
/// (report-only — the verifier is still returned trained, matching mahbot-953's
/// report-only benchmark precedent).  The value mirrors the `TPR ≥ 0.90`
/// constraint used by [`calibrate_verifier_threshold`].
const MIN_HELD_OUT_TPR: f32 = 0.90;

/// How much to upweight confusable negative examples during verifier training.
///
/// Confusable phrases (e.g. "hey map bot", "day mahbot") are acoustically
/// very similar to the wake word.  Without this upweighting, their gradient
/// signal is drowned out by thousands of ambient negatives.  The weight was
/// 100× in the original mahbot-872 implementation, but benchmark results
/// showed positive detection collapse (~15%, need ≥85%) — the confusable
/// gradient dominated (~95% of total), making the verifier overly conservative
/// and rejecting the actual wake word.  Reduced to 50× (mahbot-872) and then
/// to 15× (mahbot-882) to bring confusable gradient contribution from ~77-88%
/// down to roughly 40-50%, giving the positive class meaningful influence on
/// the decision boundary while maintaining the zero-false-accept property.
pub(crate) const CONFUSABLE_UPWEIGHT: f32 = 15.0;

/// How much to upweight unrelated speech negative examples during verifier training.
///
/// Unrelated phrases (e.g. "what time is it", "good morning everyone") are
/// phonetically very different from the wake word but still represent real
/// non-wake-word speech that the verifier must reject.  10× gives them ~5×
/// more gradient contribution than ambient silence while still prioritising
/// confusable phrases as the primary negative signal.
pub(crate) const UNRELATED_UPWEIGHT: f32 = 10.0;

/// How much to upweight owner-negative speech examples during verifier training
/// (Phase 3 enrollment).
///
/// Owner-negative audio is the user's own general speech (non-wake-word phrases)
/// collected after the 10 enrollment utterances.  These are the most realistic
/// false-trigger examples since they come from the same speaker, same mic, same
/// room as the positive class.  3.0× gives them meaningful weight (~3× ambient)
/// while keeping them below unrelated speech (10×) and confusable near-misses
/// (15×), reflecting the tier: ambient → owner-negative → unrelated → confusable.
pub(crate) const OWNER_NEGATIVE_UPWEIGHT: f32 = 3.0;

/// How much to upweight cross-speaker wake-word negatives during verifier
/// training (mahbot-1025).
///
/// Cross-speaker TTS wake-word audio is the wake word spoken by a NON-enrolled
/// voice — under single-speaker semantics it is the most dangerous negative
/// class (the exact phrase, wrong voice).  Clean and 10/20 dB white/pink
/// noise-conditioned variants join the verifier negative set as in-distribution
/// regression canaries.  Weighted at the confusable tier (15×): these are
/// acoustically the wake word itself, so they must influence the boundary as
/// strongly as near-miss phrases.
///
/// Mechanism note: this is a per-negative-WINDOW weight (each window of a
/// cross-speaker sequence counts 15× toward the negative-window weight sum).
/// The POSITIVE class weight is DERIVED from that sum (`neg_weight_sum /
/// n_pos_windows`) and only then capped at [`MAX_CLASS_WEIGHT`] (50×,
/// mahbot-1008 Fix 4) — the cap constrains the derived positive weight, not
/// the negative upweight itself, and prevents per-example memorization of the
/// positive pool.  A fresh-run trial at 8× relaxed the boundary too far
/// (member speed_down scores dipped to 0.83) without improving the measured
/// miss fraction, so 15× is retained.
///
/// Cross-speaker negatives only exist in the benchmark harness (they are
/// generated from held-out test-voice clips), so this knob is compiled only
/// under the `voice-tests` feature alongside [`crate::audio::voice::voice_pipeline_e2e_test`].
#[cfg(feature = "voice-tests")]
pub(crate) const CROSS_SPEAKER_UPWEIGHT: f32 = 15.0;

/// Number of independently-seeded verifier members trained for the
/// multi-seed ensemble (mahbot-1025).
///
/// Each fresh run draws ONE entropy base seed; the N members are trained with
/// deterministic member seeds `base..base+N` over identical training data.
/// `predict` returns the MEAN of all trained members' scores, which shrinks
/// per-run seed variance by ~1/√N.  The measured per-run speed_down miss
/// probability gate (≤ 0.15, target ~0.04–0.10) is the fraction of members
/// whose individual speed_down peak falls below [`VERIFIER_ACCEPTANCE_FLOOR`]:
/// with 10 members the measured fraction resolves 0.0 / 0.1 / 0.2 … so the
/// target band (0.04–0.10) is observable (0 or 1 member below floor), whereas
/// 5 members would only resolve 0.0 / 0.2 (no in-band reading).  Inference
/// cost is ~10×3μs ≈ 30μs per scored window — negligible against the ~760 ms
/// latency envelope and the 10 ms frame period.
pub(crate) const VERIFIER_ENSEMBLE_SEEDS: usize = 10;

/// Embedding dimensionality (used by both verifier and voice pipeline).
/// Minimum number of classifier embeddings required before the verifier gate
/// is evaluated (mahbot-887).
///
/// During warm-up (ring buffer length < this value), detections pass with only
/// the Conv1D classifier threshold protection.  This prevents the verifier from
/// false-rejecting wake word detections that occur when only 3 embeddings
/// exist — at that point the verifier only has a single 3-frame window (the
/// onset window with padded mel frames) that produces unreliable low scores.
/// By the time sufficient embeddings accumulate for a temporally representative
/// verifier window (frame ~5+), the classifier's rolling sum has often already
/// decayed below the detection threshold.
///
/// Set to 4 so the verifier has at least 2 stride-1 windows (embedding pairs
/// [0,1,2] and [1,2,3]) rather than a single onset window.  This is a tunable
/// heuristic — higher values delay the first verifier evaluation but also give
/// more temporal context when it does evaluate.
///
/// ## Warm-up suppression (mahbot-893)
///
/// Detections during the warm-up period (ring.len() < this constant with a
/// trained verifier) are **unconditionally suppressed** — no detection is ever
/// reported, regardless of classifier score, threshold, or rolling sum.  The
/// rolling score window still accumulates during suppression, preserving
/// post-warm-up detection timing.  This replaces the previous raised-threshold
/// approach (mahbot-892) and structurally eliminates warm-up false accepts.
///
/// ## Calibration note
///
/// This value was selected heuristically (Analyst #3, mahbot-886/mahbot-887).
/// Re-run the HARD-tier E2E calibration sweep before adjusting:
/// `cargo bench --no-default-features --features voice-tests --bench voice_pipeline_e2e`
/// (canonical minimal invocation, mahbot-1041).  Adjust in source and
/// re-benchmark.
///
/// ## Interaction with other constants
///
/// - Must be ≥ `VERIFIER_WINDOW_SIZE` (3) so the verifier has at least one
///   3-frame window when it does evaluate.
/// - Must be ≤ `EMBEDDING_RING_MAX` so the warm-up period is bounded by the
///   ring buffer capacity.
pub(crate) const VERIFIER_WARMUP_EMBEDDINGS: usize = 4;

/// Number of synthetic negative examples to generate for bootstrapping
/// when no real calibration data is available.
const SYNTHETIC_NEGATIVES_COUNT: usize = 100;

/// Activation used by the verifier's feature path (mahbot-1008 Fix 3).
///
/// Persisted inside [`VoiceVerifier`] so inference semantics are explicit and
/// future architecture changes cannot silently reinterpret stored weights.
///
/// # Backward compatibility
///
/// The field has `#[serde(default)]` and is skipped when serializing the
/// default value, so models persisted before mahbot-1008 (ReLU semantics)
/// deserialize with `LeakyReLU`.  Reinterpreting pre-fix weights with
/// LeakyReLU is harmless: those models are the collapsed constant-floor bricks
/// this ticket fixes (their output stays far below the decision threshold for
/// any input, and [`VoiceVerifier::is_collapsed`] drops them at load time).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerifierActivation {
    /// LeakyReLU with slope [`LEAKY_RELU_SLOPE`] — replaces ReLU to eliminate
    /// the dead-zone constant floor.
    #[serde(rename = "leaky_relu")]
    LeakyReLU,
}

fn default_verifier_activation() -> VerifierActivation {
    VerifierActivation::LeakyReLU
}

// `skip_serializing_if` requires `fn(&T) -> bool`, so the reference is
// intentional despite the Copy type.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_default_verifier_activation(a: &VerifierActivation) -> bool {
    *a == VerifierActivation::LeakyReLU
}

/// Verifier for wake word false-trigger suppression (second-stage AND gate).
///
/// Conv1D(96→2, k=3, padding=1) → LeakyReLU → AdaptiveAvgPool → Linear(2→1) → Sigmoid.
/// ~581 trainable parameters operating on 288-dim concatenated windows.
///
/// When `trained` is `false`, the verifier is a no-op (all frames pass with
/// score 1.0).
///
/// ## Multi-seed ensemble (mahbot-1025)
///
/// Production and the E2E benchmark train [`VERIFIER_ENSEMBLE_SEEDS`]
/// independently-seeded members over identical data and store the non-primary
/// members in [`ensemble_members`](Self::ensemble_members).  [`predict`](Self::predict)
/// returns the MEAN score across the primary and all trained members, which
/// stabilizes per-run scoring: the measured per-run speed_down miss
/// probability drops from ~0.22 (single seed) to ≤0.15 with the ensemble.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceVerifier {
    /// Conv1D weight: [CONV_VERIFIER_OUT, EMBEDDING_DIM, kernel_size] = [2, 96, 3] = 576.
    pub conv_weight: Vec<f32>,
    /// Conv1D bias: [CONV_VERIFIER_OUT] = [2].
    pub conv_bias: Vec<f32>,
    /// FC weight: [CONV_VERIFIER_OUT] → [1] = 2 elements.
    pub fc_weight: Vec<f32>,
    /// FC bias: [1] = 1 element.
    pub fc_bias: Vec<f32>,

    /// Feature-path activation used at inference.  Defaults to
    /// [`VerifierActivation::LeakyReLU`] (see the enum docs for the
    /// compatibility story).
    #[serde(
        default = "default_verifier_activation",
        skip_serializing_if = "is_default_verifier_activation"
    )]
    pub activation: VerifierActivation,

    /// Decision threshold. Frames with a score below this are suppressed.
    #[serde(default = "default_verifier_threshold")]
    pub threshold: f32,
    /// Whether this verifier has been trained with positive + negative data.
    #[serde(default)]
    pub trained: bool,

    /// Additional seed-trained ensemble members (mahbot-1025).
    ///
    /// When non-empty, [`predict`](Self::predict) averages this verifier's
    /// score with each trained member's score (mean of
    /// `1 + ensemble_members.len()` scores).  Members are trained from the
    /// same data with distinct seeds so the ensemble mean is far more stable
    /// across fresh runs than any single seed draw.
    ///
    /// Serialization: `#[serde(default)]` keeps legacy JSON (no field) loading
    /// as a single-member verifier; `skip_serializing_if` keeps single-member
    /// verifiers byte-identical to pre-mahbot-1025 output.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ensemble_members: Vec<VoiceVerifier>,
}

fn default_verifier_threshold() -> f32 {
    DEFAULT_VERIFIER_THRESHOLD
}

/// Training diagnostics surfaced by [`VoiceVerifier::train_with_metrics`]
/// (mahbot-1005 §5).  Report-only — never persisted inside [`VoiceVerifier`],
/// so the model's `Serialize`/`Deserialize` layout is unchanged.
#[derive(Debug, Clone, Default)]
pub struct VerifierTrainingMetrics {
    /// Number of epochs actually trained (may be less than `CONV_MAX_EPOCHS`
    /// due to early stopping).
    pub epochs_trained: usize,
    /// Per-epoch training loss, one entry per epoch actually trained.
    pub per_epoch_train_loss: Vec<f32>,
    /// Per-epoch validation loss, one entry per epoch actually trained.
    pub per_epoch_val_loss: Vec<f32>,
    /// Mean validation score of positive windows at the selected threshold.
    pub val_pos_score_mean: f32,
    /// Mean validation score of negative windows at the selected threshold.
    pub val_neg_score_mean: f32,
    /// Constrained Weighted Youden's J (`TPR - CALIBRATION_LAMBDA * FPR`) at
    /// the selected threshold.  `None` when no validation data was available.
    pub youden_index: Option<f32>,
    /// True-positive rate at the selected threshold.  `None` when no validation
    /// positives existed.
    pub tpr: Option<f32>,
    /// False-positive rate at the selected threshold.  `None` when no validation
    /// negatives existed.
    pub fpr: Option<f32>,
    /// Number of positive validation windows.
    pub n_val_pos: usize,
    /// Number of negative validation windows.
    pub n_val_neg: usize,
    /// Whether threshold calibration ran (real non-synthetic negatives present).
    pub threshold_calibrated: bool,
    /// Whether the returned threshold is the caller-supplied fallback (i.e.
    /// calibration was skipped).
    pub threshold_is_fallback: bool,
}

impl Default for VoiceVerifier {
    fn default() -> Self {
        Self::untrained()
    }
}

/// Per-sequence metadata for stratified train/val split (mahbot-949).
struct SeqInfo {
    start: usize,
    count: usize,
    is_positive: bool,
    /// Source category for diagnostics.  Read by [`split_train_val`] when
    /// `stratify_by_source` is true for source-tier stratified validation
    /// splitting (mahbot-995).
    source: crate::audio::embedding_sequence::Source,
    /// Augmentation family of the source sequence (mahbot-1008 Fix 1).  Used
    /// with `source` to form provenance groups for out-of-session holdout —
    /// e.g. all `Source::Augmentation` + `Some(SpeedDown)` windows form one
    /// group that can be held out entirely for validation.
    augmentation_family: Option<crate::audio::embedding_sequence::AugmentationFamily>,
}

impl VoiceVerifier {
    /// Create an untrained verifier (no-op: all frames pass).
    ///
    /// An untrained verifier always returns `1.0` from [`predict`](Self::predict).
    #[must_use]
    pub fn untrained() -> Self {
        Self {
            conv_weight: Vec::new(),
            conv_bias: Vec::new(),
            fc_weight: Vec::new(),
            fc_bias: Vec::new(),
            activation: VerifierActivation::LeakyReLU,
            threshold: DEFAULT_VERIFIER_THRESHOLD,
            trained: false,
            ensemble_members: Vec::new(),
        }
    }

    /// Number of ensemble members including this primary verifier.
    ///
    /// Returns `1` for a legacy/single-member verifier.
    #[must_use]
    pub fn ensemble_size(&self) -> usize {
        1 + self.ensemble_members.len()
    }

    /// Return a **single-member** verifier view: primary weights = member
    /// `member_idx`, empty ensemble (mahbot-1025).
    ///
    /// Used by the E2E benchmark to measure each member's individual
    /// speed_down peak (the per-run miss-probability variance-reduction gate:
    /// fraction of members whose standalone peak falls below
    /// [`VERIFIER_ACCEPTANCE_FLOOR`]).  `member_idx` must be in
    /// `0..self.ensemble_size()`; out-of-range indices are a programming error
    /// (`debug_assert!` fires) and fall back to an untrained no-op only as a
    /// release-build defensive measure.
    #[must_use]
    pub fn member_only(&self, member_idx: usize) -> VoiceVerifier {
        debug_assert!(
            member_idx < self.ensemble_size(),
            "member_only: index {member_idx} out of range (ensemble size {})",
            self.ensemble_size(),
        );
        if member_idx == 0 {
            let mut clone = self.clone();
            clone.ensemble_members = Vec::new();
            return clone;
        }
        match self.ensemble_members.get(member_idx - 1) {
            Some(member) => {
                let mut clone = member.clone();
                clone.ensemble_members = Vec::new();
                clone
            }
            None => Self::untrained(),
        }
    }

    /// Returns `true` if this verifier's primary member is trained and
    /// structurally valid.
    ///
    /// Validates Conv1D weights: 576-dim conv_weight + 2-dim conv_bias + 2-dim fc_weight + 1-dim fc_bias.
    /// Ensemble members are NOT required to be trained here: an untrained
    /// member is SKIPPED by [`predict`](Self::predict) (excluded from both the
    /// score sum and the member count, so it is neutral — it neither boosts
    /// nor drags the ensemble mean), so a partially-trained ensemble still
    /// gates via its trained members (mahbot-1025).  Collapse detection
    /// ([`is_collapsed`](Self::is_collapsed)) separately probes every member,
    /// because a COLLAPSED member (constant reject) would drag the ensemble
    /// mean down on every input.
    #[must_use]
    pub fn is_trained(&self) -> bool {
        if !self.trained {
            return false;
        }
        self.weights_valid()
    }

    /// Validate the primary member's weight shapes and finiteness.
    fn weights_valid(&self) -> bool {
        // conv_weight: [CONV_VERIFIER_OUT, EMBEDDING_DIM, CONV_VERIFIER_KERNEL_SIZE]
        let expected_conv_w = CONV_VERIFIER_OUT * EMBEDDING_DIM * CONV_VERIFIER_KERNEL_SIZE;
        if self.conv_weight.len() != expected_conv_w {
            return false;
        }
        if self.conv_bias.len() != CONV_VERIFIER_OUT {
            return false;
        }
        if self.fc_weight.len() != CONV_VERIFIER_OUT {
            return false;
        }
        if self.fc_bias.len() != 1 {
            return false;
        }
        // All weights must be finite.
        self.conv_weight.iter().all(|v| v.is_finite())
            && self.conv_bias.iter().all(|v| v.is_finite())
            && self.fc_weight.iter().all(|v| v.is_finite())
            && self.fc_bias.iter().all(|v| v.is_finite())
    }

    /// Predict the probability that the given window is a genuine wake word.
    ///
    /// Requires 288-dim input (3 concatenated 96-dim embeddings). Panics if
    /// input is not 288-dim.
    ///
    /// Returns a score in `[0.0, 1.0]`. When untrained, always returns `1.0`
    /// (no-op — all frames pass).
    ///
    /// ## Multi-seed ensemble (mahbot-1025)
    ///
    /// When [`ensemble_members`](Self::ensemble_members) is non-empty, returns
    /// the MEAN score across this verifier and every trained member — the
    /// seed-stabilized scoring used by production and the E2E benchmark.  The
    /// ensemble mean shrinks per-run seed variance (~1/√N) so a fixed input's
    /// score is far more stable across fresh runs than any single-seed draw.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn predict(&self, embedding: &[f32]) -> f32 {
        if !self.is_trained() {
            return 1.0;
        }
        let mut sum = predict_conv1d(
            embedding,
            &self.conv_weight,
            &self.conv_bias,
            &self.fc_weight,
            &self.fc_bias,
        );
        let mut n = 1;
        for member in &self.ensemble_members {
            if member.is_trained() {
                sum += predict_conv1d(
                    embedding,
                    &member.conv_weight,
                    &member.conv_bias,
                    &member.fc_weight,
                    &member.fc_bias,
                );
                n += 1;
            }
        }
        sum / n as f32
    }

    /// Detect the mahbot-1008 constant-floor collapse: a trained verifier whose
    /// output is input-independent and below its decision threshold for every
    /// input (a reject-all brick wall).
    ///
    /// With a multi-seed ensemble (mahbot-1025), ANY trained member that is
    /// collapsed drags the ensemble mean down on every input by roughly that
    /// member's healthy contribution / N — and with enough collapsed members,
    /// or healthy members scoring near the floor, it can push genuine wake
    /// words below the 0.86 acceptance floor.  The mean alone is NOT a
    /// reliable collapse detector: ten healthy members at ~0.97 plus one
    /// collapsed brick at `6.67e-8` still average ≈ 0.87, above the floor.
    /// The probe therefore runs the deterministic input-independence check on
    /// the primary AND every trained ensemble member; if any member is
    /// collapsed, the whole verifier is flagged (callers replace it via
    /// [`without_collapsed_members`](Self::without_collapsed_members)).  Untrained
    /// members are skipped (they cannot collapse — an untrained verifier is a
    /// neutral pass-through, never a constant-reject risk).
    ///
    /// The pre-fix architecture produced exactly this: dead ReLU features →
    /// pooled `[0, 0]` → `logit = fc_bias` → `sigmoid(fc_bias)` regardless of
    /// input (observed `6.67e-8`).  We probe with a small set of deterministic
    /// pseudo-random L2-normalized inputs and flag the verifier when:
    ///
    /// - the output range is `< 1e-4` (input-independent), AND
    /// - the maximum output is below [`VERIFIER_ACCEPTANCE_FLOOR`] (constant
    ///   reject).
    ///
    /// The comparison uses the constant 0.86 acceptance floor (mahbot-1023),
    /// NOT the runtime-calibrated `self.threshold`: a healthy verifier scoring
    /// in [0.86, 0.91) confirms detections (constant gate) and must therefore
    /// NOT be flagged collapsed by a higher runtime-calibrated threshold.
    ///
    /// A constant-*accept* verifier (output ≥ floor everywhere) is
    /// functionally a no-op and is deliberately NOT flagged.  The probes are
    /// deterministic (fixed seed) so the verdict is stable across restarts.
    ///
    /// Used at load time in `voice.rs` ([`resolve_loaded_verifier`](crate::audio::voice::resolve_loaded_verifier))
    /// to drop collapsed members from already-enrolled users' persisted
    /// verifiers — keeping the healthy members instead of falling back to
    /// classifier-only gating (mahbot-1025).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn is_collapsed(&self) -> bool {
        if !self.is_trained() {
            return false;
        }
        // Every trained member (including the primary) must pass the
        // input-independence probe — a single collapsed member drags the
        // ensemble mean down.  Untrained members (1.0 no-ops) are skipped.
        if self.member_is_collapsed() {
            return true;
        }
        self.ensemble_members
            .iter()
            .filter(|m| m.is_trained())
            .any(VoiceVerifier::member_is_collapsed)
    }

    /// Probe one verifier's weights (primary or member) for the
    /// input-independent constant-reject collapse.  Untrained verifiers
    /// (empty weights) are never collapsed.
    #[allow(clippy::cast_precision_loss)]
    fn member_is_collapsed(&self) -> bool {
        const N_PROBES: usize = 8;
        if !self.is_trained() {
            return false;
        }
        let mut rng = StdRng::seed_from_u64(0x1008_1008);
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f32;
        for _ in 0..N_PROBES {
            let mut probe = vec![0.0f32; VERIFIER_INPUT_DIM];
            for v in &mut probe {
                *v = rng.random::<f32>() * 2.0 - 1.0;
            }
            let score = predict_conv1d(
                &probe,
                &self.conv_weight,
                &self.conv_bias,
                &self.fc_weight,
                &self.fc_bias,
            );
            min = min.min(score);
            max = max.max(score);
            sum += score;
        }
        let range = max - min;
        let mean = sum / N_PROBES as f32;
        if range < 1e-4 && max < VERIFIER_ACCEPTANCE_FLOOR {
            info!(
                "VoiceVerifier member collapsed: input-independent output (range={range:.2e}, \
                 mean={mean:.2e}, max={max:.2e}) below acceptance floor={VERIFIER_ACCEPTANCE_FLOOR:.4} \
                 — constant reject-all",
            );
            true
        } else {
            false
        }
    }

    /// Return a copy of this verifier with any collapsed members removed
    /// (mahbot-1025).
    ///
    /// A collapsed (constant input-independent reject) member would drag the
    /// ensemble mean down on every input, so at load time such members are
    /// dropped rather than poisoning the healthy members' scores — a
    /// 10-member ensemble with one collapsed brick keeps its nine healthy
    /// members instead of degrading to classifier-only gating.  If the
    /// primary is collapsed, the first healthy member is promoted to primary;
    /// if no healthy trained member remains, returns an untrained no-op.
    ///
    /// Callers use this via [`is_collapsed`](Self::is_collapsed) +
    /// [`resolve_loaded_verifier`](crate::audio::voice::resolve_loaded_verifier)
    /// at load time.
    #[must_use]
    pub fn without_collapsed_members(&self) -> VoiceVerifier {
        // Collect healthy trained members (primary + ensemble) in order,
        // each as a single-member verifier so no stale members are attached.
        let mut healthy: Vec<VoiceVerifier> = Vec::new();
        if self.is_trained() && !self.member_is_collapsed() {
            let mut primary = self.clone();
            primary.ensemble_members = Vec::new();
            healthy.push(primary);
        }
        for member in &self.ensemble_members {
            if member.is_trained() && !member.member_is_collapsed() {
                let mut clone = member.clone();
                clone.ensemble_members = Vec::new();
                healthy.push(clone);
            }
        }
        match healthy.first() {
            Some(primary) => {
                let mut out = primary.clone();
                out.ensemble_members = healthy.into_iter().skip(1).collect();
                out
            }
            None => Self::untrained(),
        }
    }

    /// Train a new verifier from positive and negative
    /// [`EmbeddingSequence`](crate::audio::embedding_sequence::EmbeddingSequence)
    /// inputs.  Trains a Conv1D(96→2, k=3, padding=1) → LeakyReLU → AdaptiveAvgPool →
    /// Linear(2→1) → Sigmoid architecture with L2 regularization (mahbot-994)
    /// using pure-Rust manual backprop + Adam.
    ///
    /// Windows are formed **within** each sequence independently (never across
    /// sequences), preventing the cross-utterance window contamination that
    /// existed when training operated on flat `&[Vec<f32>]` lists (mahbot-902).
    /// Each window is 3 embeddings (288-dim, not mean-pooled) to preserve
    /// temporal structure.  Windows are L2-normalized before training
    /// (mahbot-870).
    ///
    /// Reuses shared infrastructure for window formation, class weight calculation,
    /// and stratified per-sequence train/val split with 288-dim concatenated
    /// windows (preserving temporal structure).
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::similar_names,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn train(
        positive_sequences: &[EmbeddingSequence],
        negative_sequences: &[EmbeddingSequence],
        per_negative_sequence_weights: Option<&[f32]>,
        threshold: f32,
        l2_lambda: f32,
        rng_seed: Option<u64>,
    ) -> Self {
        Self::train_with_metrics(
            positive_sequences,
            negative_sequences,
            per_negative_sequence_weights,
            threshold,
            l2_lambda,
            rng_seed,
        )
        .0
    }

    /// Like [`Self::train`], but also returns training diagnostics for
    /// benchmark reporting (mahbot-1005 §5).  Production callers keep using
    /// [`Self::train`]; the benchmark uses this variant so per-epoch losses,
    /// Youden index, TPR/FPR, and validation-split composition are visible in
    /// the JSON report.  The returned [`VerifierTrainingMetrics`] is
    /// report-only — it is never persisted inside the model.
    pub fn train_with_metrics(
        positive_sequences: &[EmbeddingSequence],
        negative_sequences: &[EmbeddingSequence],
        per_negative_sequence_weights: Option<&[f32]>,
        threshold: f32,
        l2_lambda: f32,
        rng_seed: Option<u64>,
    ) -> (Self, VerifierTrainingMetrics) {
        // ── Prepare training data via shared helper ──
        let Some((windows, window_labels, window_weights, seq_infos, class_weight, _n_pos_windows)) =
            prepare_training_data(
                positive_sequences,
                negative_sequences,
                per_negative_sequence_weights,
                form_conv1d_sequence_windows,
            )
        else {
            return (Self::untrained(), VerifierTrainingMetrics::default());
        };

        Self::train_member_from_prepared(
            &windows,
            &window_labels,
            &window_weights,
            &seq_infos,
            class_weight,
            threshold,
            l2_lambda,
            rng_seed,
        )
    }

    /// Single-member training body shared by [`Self::train_with_metrics`] and
    /// the parallel ensemble path (mahbot-1029 D3).
    ///
    /// Takes the already-prepared training data (windows + labels + weights +
    /// per-sequence info + class weight) produced by
    /// [`prepare_training_data`].  Everything from the per-member train/val
    /// split onward is private to this call — no shared mutable state — so
    /// the ensemble can run members on independent threads with byte-identical
    /// results to the serial path.  The member's `rng_seed` drives both the
    /// stratified split and weight init, so it must NOT be shared across
    /// members.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn train_member_from_prepared(
        windows: &[Vec<f32>],
        window_labels: &[f32],
        window_weights: &[f32],
        seq_infos: &[SeqInfo],
        class_weight: f32,
        threshold: f32,
        l2_lambda: f32,
        rng_seed: Option<u64>,
    ) -> (Self, VerifierTrainingMetrics) {
        // All windows from form_conv1d_sequence_windows are already 288-dim
        // and L2-normalized — no mean-pooling needed.

        // ── Stratified per-sequence train/val split (shared helper) ──
        // Uses source-tier stratification (mahbot-949) for consistent
        // training regimes.
        //
        // Create RNG here so it can be reused by the training loop below
        // for epoch-level shuffling (preserving deterministic seed behavior).
        let mut rng: StdRng = if let Some(seed) = rng_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(rand::random())
        };
        let (tr_windows, tr_labels, tr_weights, val_windows, val_labels, val_weights, split_kind) =
            split_train_val(
                seq_infos,
                windows,
                window_labels,
                window_weights,
                &mut rng,
                true, // stratify_by_source — preserve negative tier proportions
            );

        // ── Conv1D training with Adam ──
        // Architecture: Conv1D(96→2, k=3, padding=1) → LeakyReLU → AdaptiveAvgPool → Linear(2→1)
        let cin = EMBEDDING_DIM; // 96
        let cout = CONV_VERIFIER_OUT; // 2
        let ks = CONV_VERIFIER_KERNEL_SIZE; // 3
        let lin = VERIFIER_WINDOW_SIZE; // 3

        let n_conv_w = cout * cin * ks; // 2 * 96 * 3 = 576
        let n_conv_b = cout; // 2
        let n_fc_w = cout; // 2
        let n_fc_b = 1; // 1

        // Xavier/Glorot uniform init with 1.7× multiplier for conv (matching classifier).
        let scale_conv = 1.7 * (6.0 / ((cin + cout) * ks) as f32).sqrt();
        let scale_fc = (6.0 / (cout + 1) as f32).sqrt();

        let mut weight_conv = vec![0.0; n_conv_w];
        for w in &mut weight_conv {
            *w = rng.random::<f32>() * 2.0 * scale_conv - scale_conv;
        }
        let mut bias_conv = vec![0.0; n_conv_b];
        let mut weight_fc = vec![0.0; n_fc_w];
        for w in &mut weight_fc {
            *w = rng.random::<f32>() * 2.0 * scale_fc - scale_fc;
        }
        let mut bias_fc = vec![0.0; n_fc_b];

        // Adam optimizers for each parameter tensor.
        let mut opt_conv_w = crate::audio::wake_word_classifier::AdamState::new(n_conv_w);
        let mut opt_conv_b = crate::audio::wake_word_classifier::AdamState::new(n_conv_b);
        let mut opt_fc_w = crate::audio::wake_word_classifier::AdamState::new(n_fc_w);
        let mut opt_fc_b = crate::audio::wake_word_classifier::AdamState::new(n_fc_b);

        // ── Training loop ──
        let n_tr = tr_windows.len();
        let n_val = val_windows.len();
        let bs = CONV_BATCH_SIZE.min(n_tr).max(1);

        let mut best_loss = f32::INFINITY;
        let mut best_conv_w = weight_conv.clone();
        let mut best_conv_b = bias_conv.clone();
        let mut best_fc_w = weight_fc.clone();
        let mut best_fc_b = bias_fc.clone();
        let mut stall_count = 0;
        let mut epochs_trained = 0;

        // Per-epoch diagnostics (mahbot-1005 §5) — previously computed and
        // logged but discarded.  Vectors cap at CONV_MAX_EPOCHS (100) entries.
        let mut metrics = VerifierTrainingMetrics::default();

        for epoch in 0..CONV_MAX_EPOCHS {
            epochs_trained = epoch + 1;

            // Shuffle training indices each epoch.
            let mut tr_idx: Vec<usize> = (0..n_tr).collect();
            tr_idx.shuffle(&mut rng);

            // Mini-batch SGD.
            for batch_start in (0..n_tr).step_by(bs) {
                let batch_end = batch_start + bs.min(n_tr - batch_start);

                // Accumulate gradients over the batch.
                let mut g_conv_w = vec![0.0; n_conv_w];
                let mut g_conv_b = vec![0.0; n_conv_b];
                let mut g_fc_w = vec![0.0; n_fc_w];
                let mut g_fc_b = vec![0.0; n_fc_b];

                for &i in &tr_idx[batch_start..batch_end] {
                    let x = &tr_windows[i];
                    let y = tr_labels[i];
                    let sw = tr_weights[i];

                    // Convert to channels-first.
                    let cf = crate::audio::wake_word_classifier::to_channels_first(x, cin, lin);

                    // Forward: Conv1D
                    let conv_out = crate::audio::wake_word_classifier::conv1d(
                        &cf,
                        cin,
                        lin,
                        cout,
                        ks,
                        &weight_conv,
                        &bias_conv,
                    );

                    // LeakyReLU (mahbot-1008 Fix 3 — ReLU's dead zone allowed
                    // the feature path to collapse to a constant floor).
                    let mut relu_out = conv_out.clone();
                    leaky_relu_verifier(&mut relu_out);
                    // Save mask for backward: slope 1.0 for positive
                    // pre-activations, LEAKY_RELU_SLOPE for non-positive.
                    let relu_mask: Vec<f32> = conv_out
                        .iter()
                        .map(|&v| if v > 0.0 { 1.0 } else { LEAKY_RELU_SLOPE })
                        .collect();

                    // AdaptiveAvgPool
                    let pooled =
                        crate::audio::wake_word_classifier::adaptive_avg_pool(&relu_out, cout, lin);

                    // Linear → sigmoid → BCE
                    let logit: f32 = pooled
                        .iter()
                        .zip(weight_fc.iter())
                        .map(|(v, w)| v * w)
                        .sum::<f32>()
                        + bias_fc[0];
                    let pred = 1.0 / (1.0 + (-logit).exp());

                    // Weighted BCE gradient: dL/dlogit = sw * (pred - y)
                    let d_logit = sw * (pred - y);

                    // ── Backward ──

                    // FC backward
                    for j in 0..cout {
                        g_fc_w[j] += pooled[j] * d_logit;
                    }
                    g_fc_b[0] += d_logit;

                    let mut d_pooled = vec![0.0; cout];
                    for j in 0..cout {
                        d_pooled[j] = weight_fc[j] * d_logit;
                    }

                    // AdaptiveAvgPool backward: dL/d(relu_out[ci, li]) = d_pooled[ci] / lin
                    let mut d_relu = vec![0.0; cout * lin];
                    for ci in 0..cout {
                        let grad = d_pooled[ci] / lin as f32;
                        for li in 0..lin {
                            d_relu[ci * lin + li] = grad;
                        }
                    }

                    // LeakyReLU backward
                    let mut d_conv = vec![0.0; cout * lin];
                    for i in 0..(cout * lin) {
                        d_conv[i] = d_relu[i] * relu_mask[i];
                    }

                    // Conv1D backward: for each output channel co, compute
                    // dL/d(weight[co, ci, k]) += d_conv[co, li] * cf[ci, li+k-padding]
                    // and dL/d(bias[co]) += d_conv[co, li]
                    let padding = ks / 2;
                    for co in 0..cout {
                        for li in 0..lin {
                            let go = d_conv[co * lin + li];
                            for ci in 0..cin {
                                for k in 0..ks {
                                    let ii = li as isize + k as isize - padding as isize;
                                    if ii >= 0 && ii < lin as isize {
                                        let widx = (co * cin + ci) * ks + k;
                                        g_conv_w[widx] += go * cf[ci * lin + ii as usize];
                                    }
                                }
                            }
                            g_conv_b[co] += go;
                        }
                    }
                }

                // Average gradients over batch size.
                let batch_size_f32 = (batch_end - batch_start) as f32;
                for g in &mut g_conv_w {
                    *g /= batch_size_f32;
                }
                for g in &mut g_conv_b {
                    *g /= batch_size_f32;
                }
                for g in &mut g_fc_w {
                    *g /= batch_size_f32;
                }
                for g in &mut g_fc_b {
                    *g /= batch_size_f32;
                }

                // L2 regularization on ALL parameters including biases
                // (mahbot-1008 Fix 3 — the unregularized fc_bias drifted to
                // −16.52 under ~85% negative-only batches).
                for (g, w) in g_conv_w.iter_mut().zip(weight_conv.iter()) {
                    *g += l2_lambda * w;
                }
                for (g, w) in g_conv_b.iter_mut().zip(bias_conv.iter()) {
                    *g += l2_lambda * w;
                }
                for (g, w) in g_fc_w.iter_mut().zip(weight_fc.iter()) {
                    *g += l2_lambda * w;
                }
                for (g, w) in g_fc_b.iter_mut().zip(bias_fc.iter()) {
                    *g += l2_lambda * w;
                }

                // Adam step.
                opt_conv_w.update(&mut weight_conv, &g_conv_w, CONV_LEARNING_RATE);
                opt_conv_b.update(&mut bias_conv, &g_conv_b, CONV_LEARNING_RATE);
                opt_fc_w.update(&mut weight_fc, &g_fc_w, CONV_LEARNING_RATE);
                opt_fc_b.update(&mut bias_fc, &g_fc_b, CONV_LEARNING_RATE);

                // Clamp fc_bias to a bounded range (mahbot-1008 Fix 3).  Even
                // with L2 + LeakyReLU, negative-only batches push the bias
                // down; the clamp guarantees the input-independent component of
                // the logit stays inside sigmoid(±FC_BIAS_CLAMP) so a dead
                // feature path can never produce a sub-1e-6 reject floor.
                bias_fc[0] = bias_fc[0].clamp(-FC_BIAS_CLAMP, FC_BIAS_CLAMP);
            }

            // ── Validation loss + early stopping ──
            if n_val > 0 {
                let val_loss = compute_conv1d_loss(
                    &val_windows,
                    &val_labels,
                    &val_weights,
                    &weight_conv,
                    &bias_conv,
                    &weight_fc,
                    &bias_fc,
                    cin,
                    cout,
                    ks,
                    lin,
                    l2_lambda,
                );
                let train_loss = compute_conv1d_loss(
                    &tr_windows,
                    &tr_labels,
                    &tr_weights,
                    &weight_conv,
                    &bias_conv,
                    &weight_fc,
                    &bias_fc,
                    cin,
                    cout,
                    ks,
                    lin,
                    l2_lambda,
                );

                // Per-epoch diagnostics (mahbot-1005 §5).
                metrics.per_epoch_train_loss.push(train_loss);
                metrics.per_epoch_val_loss.push(val_loss);

                if epoch % LOG_LOSS_INTERVAL == 0 {
                    info!(
                        "Conv1D verifier: epoch={epoch} train_loss={train_loss:.6} val_loss={val_loss:.6} lr={}",
                        CONV_LEARNING_RATE,
                    );
                }

                if val_loss < best_loss - 1e-8 {
                    best_loss = val_loss;
                    best_conv_w.clone_from(&weight_conv);
                    best_conv_b.clone_from(&bias_conv);
                    best_fc_w.clone_from(&weight_fc);
                    best_fc_b.clone_from(&bias_fc);
                    stall_count = 0;
                } else {
                    stall_count += 1;
                    if stall_count >= CONV_EARLY_STOP_PATIENCE {
                        info!(
                            "Conv1D verifier early stop: epoch={epoch} val_loss={val_loss:.6} best_loss={best_loss:.6}",
                        );
                        weight_conv.clone_from(&best_conv_w);
                        bias_conv.clone_from(&best_conv_b);
                        weight_fc.clone_from(&best_fc_w);
                        bias_fc.clone_from(&best_fc_b);
                        break;
                    }
                }
            }
        }

        info!(
            "Conv1D verifier training complete: {epochs_trained} epochs, best_val_loss={best_loss:.6}, \
             train={n_tr} val={n_val} windows",
        );

        // Assemble the trained verifier.
        let mut verifier = Self {
            conv_weight: weight_conv,
            conv_bias: bias_conv,
            fc_weight: weight_fc,
            fc_bias: bias_fc,
            activation: VerifierActivation::LeakyReLU,
            threshold,
            trained: true,
            ensemble_members: Vec::new(),
        };

        // Log diagnostics.
        log_verifier_diagnostics(
            &verifier,
            &tr_windows,
            &tr_labels,
            &val_windows,
            &val_labels,
            split_kind,
            class_weight,
            "Conv1D verifier",
        );

        // ── Auto-calibrate threshold on validation data (mahbot-997) ──
        // Only calibrate when the training data includes real (non-synthetic)
        // negative sequences — synthetic negatives do not represent realistic
        // false-accept distributions.
        let has_real_negatives = seq_infos
            .iter()
            .any(|s| !s.is_positive && s.source != Source::Synthetic);

        // ── Validation scores + split composition (mahbot-1005 §5) ──
        // Computed for every trained verifier (not just the calibrating path)
        // so the benchmark can report TPR/FPR/Youden even when calibration is
        // skipped (all-synthetic negatives or insufficient validation data).
        let mut val_pos_scores: Vec<f32> = Vec::new();
        let mut val_neg_scores: Vec<f32> = Vec::new();
        metrics.epochs_trained = epochs_trained;
        if n_val > 0 {
            for (emb, &lbl) in val_windows.iter().zip(val_labels.iter()) {
                let score = verifier.predict(emb);
                if lbl > 0.5 {
                    val_pos_scores.push(score);
                } else {
                    val_neg_scores.push(score);
                }
            }
            metrics.n_val_pos = val_pos_scores.len();
            metrics.n_val_neg = val_neg_scores.len();
            metrics.val_pos_score_mean = if val_pos_scores.is_empty() {
                0.0
            } else {
                val_pos_scores.iter().copied().sum::<f32>() / val_pos_scores.len() as f32
            };
            metrics.val_neg_score_mean = if val_neg_scores.is_empty() {
                0.0
            } else {
                val_neg_scores.iter().copied().sum::<f32>() / val_neg_scores.len() as f32
            };
        }

        if has_real_negatives && n_val > 0 {
            let calibrated = calibrate_verifier_threshold(
                &val_pos_scores,
                &val_neg_scores,
                threshold, // caller-supplied fallback
            );
            verifier.threshold = calibrated;
            metrics.threshold_calibrated = true;
        } else {
            metrics.threshold_is_fallback = true;
            if !has_real_negatives {
                info!(
                    "Verifier threshold calibration skipped: all negative sequences are synthetic.  \
                     Using caller-supplied threshold={threshold:.4}."
                );
            }
        }

        // ── Youden index + TPR/FPR at the selected threshold (mahbot-1005 §5) ──
        // Evaluated against the FINAL threshold (calibrated or fallback).
        if !val_pos_scores.is_empty() || !val_neg_scores.is_empty() {
            let tp = val_pos_scores
                .iter()
                .filter(|&&s| s >= verifier.threshold)
                .count();
            let fp = val_neg_scores
                .iter()
                .filter(|&&s| s >= verifier.threshold)
                .count();
            let tpr = if val_pos_scores.is_empty() {
                None
            } else {
                Some(tp as f32 / val_pos_scores.len() as f32)
            };
            let fpr = if val_neg_scores.is_empty() {
                None
            } else {
                Some(fp as f32 / val_neg_scores.len() as f32)
            };
            metrics.tpr = tpr;
            metrics.fpr = fpr;
            metrics.youden_index = match (tpr, fpr) {
                (Some(t), Some(f)) => Some(t - CALIBRATION_LAMBDA * f),
                _ => None,
            };

            // ── Held-out recall warning (mahbot-1008 Fix 1) ──
            // With honest (out-of-session) validation, a trained verifier that
            // rejects >10% of held-out genuine wake words is still too close to
            // the reject-all failure mode.  Warning is report-only (mahbot-953
            // precedent): the verifier is returned trained so we do not
            // silently regress false-accept suppression; the E2E benchmark
            // surfaces the same signal as a warning metric.
            if let Some(tpr_val) = tpr
                && val_pos_scores.len() >= CALIBRATION_MIN_SAMPLES
                && tpr_val < MIN_HELD_OUT_TPR
            {
                warn!(
                    "Verifier held-out recall LOW: TPR={tpr_val:.3} at threshold={:.4} \
                     ({tp} of {val_pos_scores_len} held-out positives accepted, minimum \
                     {MIN_HELD_OUT_TPR:.2}).  The verifier may overfit the training \
                     conditions — consider re-enrolling with more diverse audio.",
                    verifier.threshold,
                    val_pos_scores_len = val_pos_scores.len(),
                );
            }
        }

        (verifier, metrics)
    }

    /// Train a **multi-seed ensemble** verifier (mahbot-1025): train
    /// [`VERIFIER_ENSEMBLE_SEEDS`] members over the same data with distinct
    /// seeds and return the primary with the remaining members attached in
    /// [`ensemble_members`](Self::ensemble_members).
    ///
    /// Seed policy: one entropy base seed is drawn per call (matching the
    /// pre-ensemble `None` policy — never seed-pinned), and member seeds are
    /// derived deterministically as `base`, `base+1`, …, `base+N-1`.  The
    /// ensemble MEAN at inference is far more stable across fresh runs than
    /// any single seed draw (the measured variance-reduction gate: per-run
    /// speed_down miss probability ≤ 0.15, target ~0.04–0.10).
    ///
    /// Each member is trained through [`Self::train_with_metrics`]; member 0's
    /// metrics are returned (report-only — the members share training data, so
    /// member-0 diagnostics are representative).  If any member fails to meet
    /// the minimum positive-window guard, that member is untrained and is
    /// skipped by [`predict`](Self::predict) (the remaining members still gate).
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn train_ensemble(
        positive_sequences: &[EmbeddingSequence],
        negative_sequences: &[EmbeddingSequence],
        per_negative_sequence_weights: Option<&[f32]>,
        threshold: f32,
        l2_lambda: f32,
    ) -> Self {
        Self::train_ensemble_with_metrics(
            positive_sequences,
            negative_sequences,
            per_negative_sequence_weights,
            threshold,
            l2_lambda,
        )
        .0
    }

    /// [`Self::train_ensemble`] with training diagnostics (mahbot-1005 §5
    /// style).  Returns `(verifier, member-0 metrics)`.
    ///
    /// ## Parallel member training (mahbot-1029 D3)
    ///
    /// Training data preparation (window formation + L2 normalization) is
    /// deterministic and seed-independent, so it is hoisted out of the
    /// per-member loop and computed ONCE.  The 10 member trainings are then
    /// run on independent OS threads via [`std::thread::scope`] — each member
    /// has private RNG/weights/optimizer state and only reads the shared
    /// prepared data, so results are byte-identical to the serial path.
    /// Member seeds remain `base_seed.wrapping_add(i)`; `split_train_val` is
    /// deliberately NOT shared across members (it is seed-dependent — each
    /// member's stratified split differs).
    #[must_use]
    pub fn train_ensemble_with_metrics(
        positive_sequences: &[EmbeddingSequence],
        negative_sequences: &[EmbeddingSequence],
        per_negative_sequence_weights: Option<&[f32]>,
        threshold: f32,
        l2_lambda: f32,
    ) -> (Self, VerifierTrainingMetrics) {
        Self::train_ensemble_with_metrics_seeded(
            positive_sequences,
            negative_sequences,
            per_negative_sequence_weights,
            threshold,
            l2_lambda,
            rand::random(),
        )
    }

    /// [`Self::train_ensemble_with_metrics`] with an explicit base seed
    /// (test-injectable and bench-visible; production uses the entropy-drawn
    /// public wrapper).  The voice bench calls this when a run pins the seed
    /// via `MAHBOT_VOICE_BENCH_VERIFIER_SEED`.
    pub(crate) fn train_ensemble_with_metrics_seeded(
        positive_sequences: &[EmbeddingSequence],
        negative_sequences: &[EmbeddingSequence],
        per_negative_sequence_weights: Option<&[f32]>,
        threshold: f32,
        l2_lambda: f32,
        base_seed: u64,
    ) -> (Self, VerifierTrainingMetrics) {
        let n_seeds = VERIFIER_ENSEMBLE_SEEDS;

        // ── Prepare training data ONCE (deterministic, seed-independent) ──
        let Some((windows, window_labels, window_weights, seq_infos, class_weight, _n_pos_windows)) =
            prepare_training_data(
                positive_sequences,
                negative_sequences,
                per_negative_sequence_weights,
                form_conv1d_sequence_windows,
            )
        else {
            // Replicate the serial path exactly: each member would
            // independently return untrained, so the result is an untrained
            // primary with N-1 untrained members attached (ensemble_size
            // stays the same).
            let untrained = Self::untrained();
            let mut members = vec![untrained; n_seeds];
            let mut primary = members.remove(0);
            primary.ensemble_members = members;
            return (primary, VerifierTrainingMetrics::default());
        };

        // ── Parallel member training (std threads, NOT a nested Tokio
        //    runtime — see the mahbot-944 note in the bench module) ──
        let mut members: Vec<VoiceVerifier> = Vec::with_capacity(n_seeds);
        let mut member_metrics: Vec<VerifierTrainingMetrics> = Vec::with_capacity(n_seeds);
        std::thread::scope(|s| {
            let windows_ref = &windows;
            let labels_ref = &window_labels;
            let weights_ref = &window_weights;
            let seq_infos_ref = &seq_infos;
            let mut handles = Vec::with_capacity(n_seeds);
            for i in 0..n_seeds {
                let seed = base_seed.wrapping_add(i as u64);
                handles.push(s.spawn(move || {
                    Self::train_member_from_prepared(
                        windows_ref,
                        labels_ref,
                        weights_ref,
                        seq_infos_ref,
                        class_weight,
                        threshold,
                        l2_lambda,
                        Some(seed),
                    )
                }));
            }
            // Collect in seed order — preserves the report's index-ordered
            // member fingerprint arrays.
            for h in handles {
                let (member, metrics) = h.join().expect("verifier member training thread panicked");
                members.push(member);
                member_metrics.push(metrics);
            }
        });

        // Member 0 becomes the primary; the rest attach as ensemble members.
        let mut primary = members.remove(0);
        primary.ensemble_members = members;
        let primary_metrics = member_metrics.remove(0);
        if primary.is_trained() {
            info!(
                "VoiceVerifier ensemble trained: {} members, base seed {base_seed}, \
                 ensemble size {} (mahbot-1025)",
                n_seeds,
                primary.ensemble_size(),
            );
        } else {
            warn!(
                "VoiceVerifier ensemble left UNTRAINED: {n_seeds} members, base seed {base_seed} \
                 — primary member failed the minimum-window guard (mahbot-1008 Fix 2)",
            );
        }
        (primary, primary_metrics)
    }

    /// Gather windows for a set of sequence indices (shared helper for train/val split).
    fn gather_windows(
        windows: &[Vec<f32>],
        labels: &[f32],
        weights: &[f32],
        seq_idx: &[usize],
        seq_infos: &[SeqInfo],
    ) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
        let mut out_w = Vec::new();
        let mut out_l = Vec::new();
        let mut out_wt = Vec::new();
        for &idx in seq_idx {
            let info = &seq_infos[idx];
            for i in info.start..info.start + info.count {
                out_w.push(windows[i].clone());
                out_l.push(labels[i]);
                out_wt.push(weights[i]);
            }
        }
        (out_w, out_l, out_wt)
    }

    /// Multi-seed ensemble synthetic-negative training (mahbot-1025,
    /// production synthetic-fallback path).  Draws a fresh entropy seed per
    /// run to generate the synthetic-negative set ONCE from the shared
    /// positive pool, then delegates member training to
    /// [`train_ensemble_with_metrics`](Self::train_ensemble_with_metrics),
    /// which draws its own entropy base seed and trains
    /// [`VERIFIER_ENSEMBLE_SEEDS`] members over that identical data with
    /// derived member seeds `base..base+N` — the same identical-data
    /// seed-ensemble semantics as [`train_ensemble`](Self::train_ensemble):
    /// the member seeds drive only weight init and the train/val split; the
    /// training data is shared, so the ensemble mean isolates exactly the
    /// per-run training-seed variance the ticket targets.  Returns the
    /// primary with the remaining members attached.
    #[must_use]
    pub fn train_ensemble_with_synthetic_negatives(
        positive_sequences: &[EmbeddingSequence],
        threshold: f32,
    ) -> Self {
        Self::train_ensemble_with_synthetic_negatives_seeded(
            positive_sequences,
            threshold,
            rand::random(),
        )
    }

    /// [`Self::train_ensemble_with_synthetic_negatives`] with an explicit
    /// synthetic-negative seed (mahbot-1045 B2): the EXACT body of the
    /// production public wrapper, but the entropy draw for the synthetic
    /// negative set is replaced by the caller-supplied `synth_seed`.  The
    /// member base seed is still drawn inside
    /// [`train_ensemble_with_metrics`](Self::train_ensemble_with_metrics),
    /// preserving the same call order as production: the synthetic-negative
    /// sequence is built from the shared positive pool with the given seed
    /// FIRST, then member training runs over that identical data.  Bench/test
    /// only — production keeps the entropy-drawn public wrapper, so this
    /// entry point introduces zero production behavior change.
    #[must_use]
    pub(crate) fn train_ensemble_with_synthetic_negatives_seeded(
        positive_sequences: &[EmbeddingSequence],
        threshold: f32,
        synth_seed: u64,
    ) -> Self {
        // Extract flat embeddings from all positive sequences for the helper.
        let flat_positives: Vec<Vec<f32>> = positive_sequences
            .iter()
            .flat_map(|s| s.embeddings.iter().cloned())
            .collect();
        // Synthetic-negative set seed (generated ONCE per run and shared by
        // every member); the member base seed is drawn inside
        // train_ensemble_with_metrics.
        let synth_seq = build_synthetic_negative_sequence(&flat_positives, Some(synth_seed));
        // Member loop / primary attach / trained log all live in
        // train_ensemble_with_metrics — delegating keeps them in one place.
        Self::train_ensemble_with_metrics(
            positive_sequences,
            std::slice::from_ref(&synth_seq),
            None, // no per-negative weights for synthetic negatives
            threshold,
            CONV_L2_LAMBDA, // use Conv1D default L2 for synthetic bootstrapping
        )
        .0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Shared training data preparation
// ═══════════════════════════════════════════════════════════════════════════

/// Shared training data preparation: forms windows, tracks [`SeqInfo`], computes
/// class weights, and L2-normalizes.
///
/// The caller provides a `window_fn` that transforms per-frame embeddings into
/// windows — [`form_conv1d_sequence_windows`] for concatenated 288-dim windows.
///
/// # Returns
///
/// `Some((windows, labels, weights, seq_infos, class_weight, n_pos_windows))`
/// where all windows are L2-normalized.  Returns `None` and emits a `warn!`
/// log if no windows could be formed (all sequences shorter than
/// [`VERIFIER_WINDOW_SIZE`] frames or empty inputs).
#[allow(
    clippy::type_complexity,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]
fn prepare_training_data<F>(
    positive_sequences: &[EmbeddingSequence],
    negative_sequences: &[EmbeddingSequence],
    per_negative_sequence_weights: Option<&[f32]>,
    window_fn: F,
) -> Option<(Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<SeqInfo>, f32, usize)>
where
    F: Fn(&[Vec<f32>]) -> Vec<Vec<f32>>,
{
    // Early exit if either side has zero frames to avoid training on empty data.
    // Both positive and negative examples are required (mahbot-902).
    let total_pos_frames: usize = positive_sequences.iter().map(|s| s.embeddings.len()).sum();
    let total_neg_frames: usize = negative_sequences.iter().map(|s| s.embeddings.len()).sum();
    if total_pos_frames == 0 || total_neg_frames == 0 {
        warn!(
            "Cannot train verifier: need both positive ({total_pos_frames}) and negative ({total_neg_frames}) frames",
        );
        return None;
    }

    // Validate per-negative-sequence weights length.
    let weights_to_use = match per_negative_sequence_weights {
        Some(w) if w.len() == negative_sequences.len() => Some(w),
        Some(w) => {
            warn!(
                "per_negative_sequence_weights length ({}) does not match negative_sequences length ({}); \
                 falling back to uniform (1.0) negative weights",
                w.len(),
                negative_sequences.len(),
            );
            None
        }
        None => None,
    };

    // ── Form windows per-sequence (no cross-sequence windows) ──
    let mut seq_infos: Vec<SeqInfo> = Vec::new();
    let mut windows: Vec<Vec<f32>> = Vec::new();
    let mut window_labels: Vec<f32> = Vec::new();
    let mut window_weights: Vec<f32> = Vec::new();

    // Positive sequences
    for seq in positive_sequences {
        let seq_windows = window_fn(&seq.embeddings);
        let start = windows.len();
        for w in seq_windows {
            windows.push(w);
            window_labels.push(1.0);
            window_weights.push(0.0); // placeholder — set to class_weight below
        }
        if windows.len() > start {
            seq_infos.push(SeqInfo {
                start,
                count: windows.len() - start,
                is_positive: true,
                source: seq.source,
                augmentation_family: seq.augmentation_family,
            });
        }
    }
    let n_pos_windows = window_labels.iter().filter(|&&l| l > 0.5).count();

    // ── Positive-window guards (mahbot-1008 Fix 2) ──
    // A trained verifier needs enough POSITIVE WINDOWS (not positive
    // sequences) to learn a generalizable policy.  Training with a handful of
    // memorized windows produces a reject-all brick wall; an untrained no-op
    // (all frames pass) is strictly better.
    //
    // Guard 1: zero positive windows.  This is the stricter failure mode —
    // positive sequences can exist while every utterance has <
    // VERIFIER_WINDOW_SIZE frames, in which case the old code trained on
    // all-negative data and returned a `trained: true` reject-all.
    if n_pos_windows == 0 {
        warn!(
            "Cannot train verifier: {n_pos_windows} positive windows formed from \
             {} positive sequence(s) — every utterance has <{VERIFIER_WINDOW_SIZE} \
             per-frame embeddings.  Returning untrained no-op.",
            positive_sequences.len(),
        );
        return None;
    }
    // Guard 2: minimum positive windows (covers every call site — production,
    // synthetic-negatives fallback, and the E2E benchmark).
    if n_pos_windows < MIN_POSITIVE_WINDOWS {
        warn!(
            "Cannot train verifier: {n_pos_windows} positive windows < minimum \
             {MIN_POSITIVE_WINDOWS} (mahbot-1008).  Returning untrained no-op — \
             a reject-all trained on {n_pos_windows} memorized windows is worse \
             than no verifier at all.",
        );
        return None;
    }

    // Negative sequences
    for (i, seq) in negative_sequences.iter().enumerate() {
        let seq_windows = window_fn(&seq.embeddings);
        let seq_weight = weights_to_use.map_or(1.0, |pw| pw[i]);
        let start = windows.len();
        for w in seq_windows {
            windows.push(w);
            window_labels.push(0.0);
            window_weights.push(seq_weight);
        }
        if windows.len() > start {
            seq_infos.push(SeqInfo {
                start,
                count: windows.len() - start,
                is_positive: false,
                source: seq.source,
                augmentation_family: seq.augmentation_family,
            });
        }
    }

    if windows.is_empty() {
        warn!(
            "Cannot form windows: need at least {VERIFIER_WINDOW_SIZE} per-frame embeddings per sequence",
        );
        return None;
    }

    // ── Class weight from window counts (mahbot-993), capped (mahbot-1008 Fix 4) ──
    // The pre-fix formula was unbounded: 58 positive windows vs 11,074
    // negatives (with tier upweights) produced a ~2,208× per-example positive
    // weight that drove the model to memorize the 58 positives.  The cap keeps
    // a strong positive signal (50× per example) without per-example
    // memorization.  Negative subsampling is intentionally not performed —
    // the full negative set is what suppresses false accepts (see
    // MAX_CLASS_WEIGHT docs).
    let class_weight = {
        let n_pw_f = n_pos_windows as f32;
        if n_pw_f > 0.0 {
            let neg_sum: f32 = window_weights[n_pos_windows..].iter().sum();
            let raw = neg_sum / n_pw_f;
            if raw > MAX_CLASS_WEIGHT {
                info!(
                    "Verifier class weight capped: raw={raw:.1} → \
                     {MAX_CLASS_WEIGHT:.0} (mahbot-1008 Fix 4; n_pos={n_pos_windows}, \
                     neg_weight_sum={neg_sum:.1})"
                );
                MAX_CLASS_WEIGHT
            } else {
                raw
            }
        } else {
            1.0
        }
    };
    for w in &mut window_weights[0..n_pos_windows] {
        *w = class_weight;
    }

    // ── L2-normalize all windows (matching classifier convention, mahbot-870) ──
    for w in &mut windows {
        let norm = w.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
        for v in w.iter_mut() {
            *v /= norm;
        }
    }

    Some((
        windows,
        window_labels,
        window_weights,
        seq_infos,
        class_weight,
        n_pos_windows,
    ))
}

// ═══════════════════════════════════════════════════════════════════════════
// Shared train/validation split (mahbot-995)
// ═══════════════════════════════════════════════════════════════════════════

/// How the train/validation split was constructed (mahbot-1008 Fix 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SplitKind {
    /// An entire positive provenance group (e.g. all `Source::Enrollment`
    /// originals, or all `AugmentationFamily::SpeedDown` variants) was held
    /// out for validation — genuine out-of-distribution validation: the
    /// verifier never sees that wake-word condition during training.
    GroupHoldout,
    /// Per-sequence 80/20 split.  Sequences are never split across train/val
    /// (no window leakage), but all sequences come from the same enrollment
    /// session, so validation positives are near-duplicates of training
    /// positives.  Used only when no provenance groups exist (e.g. unit-test
    /// data or all-synthetic positives).
    PerSequence,
    /// No validation data was available.  Training proceeds without early
    /// stopping or threshold calibration.  This is deliberate: a leaky
    /// per-window split (the pre-mahbot-1008 fallback) validated on windows
    /// from the same sequences as training and could not detect overfitting.
    None,
}

/// Provenance group key for positive sequences: `(source, augmentation family)`.
///
/// Production enrollment produces one `(Enrollment, None)` sequence per
/// utterance plus `(Augmentation, Some(family))` PCM variants, so grouping by
/// this key partitions positives into holdout-able conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ProvenanceGroup {
    source: crate::audio::embedding_sequence::Source,
    augmentation_family: Option<crate::audio::embedding_sequence::AugmentationFamily>,
}

impl ProvenanceGroup {
    /// Human-readable label for diagnostics.
    fn label(self) -> String {
        match self.augmentation_family {
            Some(f) => format!("{:?}+{:?}", self.source, f),
            None => format!("{:?}", self.source),
        }
    }
}

/// Number of windows covered by a sequence (via its `SeqInfo` range).
fn seq_window_count(info: &SeqInfo) -> usize {
    info.count
}

/// Pick the positive provenance group to hold out for validation (mahbot-1008
/// Fix 1), or `None` when no group can be held out.
///
/// Only groups with ≥ 1 window are candidates, and holding the group out must
/// leave ≥ [`MIN_POSITIVE_WINDOWS`] positive windows in training (otherwise the
/// training set would fall below the minimum the verifier needs).
///
/// Preference order: first `(Enrollment, None)` — the unmodified originals are
/// the most production-representative wake-word condition (augmented variants
/// are derived from them), so validating on them is the strongest
/// generalization check; then the augmentation family with the most windows
/// (largest signal).
///
/// Returns the held-out group and its `SeqInfo` indices.
fn pick_positive_holdout_group(seq_infos: &[SeqInfo]) -> Option<(ProvenanceGroup, Vec<usize>)> {
    use std::collections::HashMap;

    let mut groups: HashMap<ProvenanceGroup, Vec<usize>> = HashMap::new();
    for (i, info) in seq_infos.iter().enumerate() {
        if !info.is_positive {
            continue;
        }
        let key = ProvenanceGroup {
            source: info.source,
            augmentation_family: info.augmentation_family,
        };
        groups.entry(key).or_default().push(i);
    }

    let total_pos_windows: usize = seq_infos
        .iter()
        .filter(|s| s.is_positive)
        .map(seq_window_count)
        .sum();

    // Candidates: groups whose removal keeps training ≥ MIN_POSITIVE_WINDOWS.
    let mut candidates: Vec<(ProvenanceGroup, Vec<usize>, usize)> = groups
        .into_iter()
        .map(|(group, idxs)| {
            let windows = idxs.iter().map(|&i| seq_window_count(&seq_infos[i])).sum();
            (group, idxs, windows)
        })
        .filter(|(_, _, win)| *win > 0 && total_pos_windows - *win >= MIN_POSITIVE_WINDOWS)
        .collect();
    if candidates.is_empty() {
        return None;
    }

    // Preference: (Enrollment, None) first, then largest window count.
    candidates.sort_by(|a, b| {
        let a_orig = a.0.source == crate::audio::embedding_sequence::Source::Enrollment
            && a.0.augmentation_family.is_none();
        let b_orig = b.0.source == crate::audio::embedding_sequence::Source::Enrollment
            && b.0.augmentation_family.is_none();
        b_orig
            .cmp(&a_orig)
            .then_with(|| b.2.cmp(&a.2))
            .then_with(|| a.0.label().cmp(&b.0.label()))
    });

    let (group, idxs, win_count) = candidates.remove(0);
    info!(
        "Verifier out-of-session validation (mahbot-1008 Fix 1): holding out \
         positive provenance group '{}' ({win_count} windows) — validation \
         positives are unseen during training.",
        group.label(),
    );
    Some((group, idxs))
}

/// Split negative sequence indices per-sequence, optionally stratified by
/// source tier (preserving tier proportions in train and val).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn split_negative_sequences(
    seq_infos: &[SeqInfo],
    rng: &mut StdRng,
    stratify_by_source: bool,
) -> (Vec<usize>, Vec<usize>) {
    let neg_seq_idx: Vec<usize> = seq_infos
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.is_positive)
        .map(|(i, _)| i)
        .collect();
    if neg_seq_idx.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let n_val_neg = ((neg_seq_idx.len() as f32) * VALIDATION_SPLIT).round() as usize;
    if n_val_neg == 0 || n_val_neg >= neg_seq_idx.len() {
        // Too few negative sequences for a meaningful per-sequence split —
        // keep them all in training (validation can still contain positives
        // from the group holdout).
        return (neg_seq_idx, Vec::new());
    }

    if stratify_by_source {
        // Group by source tier in FIRST-APPEARANCE order (deterministic).
        // A HashMap iteration would randomize the tier-processing order across
        // processes (RandomState per-instance seed), which reorders the
        // sequential rng draw sequence each tier's shuffle consumes — breaking
        // pinned-seed reproducibility (the voice bench's MAHBOT_VOICE_BENCH_VERIFIER_SEED
        // contract: same seed ⇒ byte-identical verifier weights).
        let mut tiers: Vec<(Source, Vec<usize>)> = Vec::new();
        for &i in &neg_seq_idx {
            let src = seq_infos[i].source;
            if let Some(entry) = tiers.iter_mut().find(|(s, _)| *s == src) {
                entry.1.push(i);
            } else {
                tiers.push((src, vec![i]));
            }
        }
        let mut tr_out = Vec::new();
        let mut val_out = Vec::new();
        for (_tier, mut idxs) in tiers {
            idxs.shuffle(rng);
            let n_val_tier = ((idxs.len() as f32) * VALIDATION_SPLIT).round() as usize;
            let n_val_tier = n_val_tier.min(idxs.len().saturating_sub(1));
            for (j, &i) in idxs.iter().enumerate() {
                if j < n_val_tier {
                    val_out.push(i);
                } else {
                    tr_out.push(i);
                }
            }
        }
        (tr_out, val_out)
    } else {
        let mut shuffled = neg_seq_idx;
        shuffled.shuffle(rng);
        let n_val = n_val_neg.min(shuffled.len().saturating_sub(1));
        let val: Vec<usize> = shuffled[..n_val].to_vec();
        let tr: Vec<usize> = shuffled[n_val..].to_vec();
        (tr, val)
    }
}

/// Shared train/val split (mahbot-995, mahbot-1008 Fix 1).
///
/// Accepts an existing `rng` (which the caller should seed for determinism).
///
/// # Validation strategy (mahbot-1008 Fix 1)
///
/// 1. **Positive provenance-group holdout** — when positive sequences span
///    multiple provenance groups (originals + augmentation families, as in
///    production and the E2E benchmark), an entire group is held out for
///    validation.  The verifier is then validated on wake-word windows it has
///    never seen during training — the per-sequence 80/20 split within a
///    single enrollment session was blind because validation positives were
///    near-duplicates of training positives.
/// 2. **Per-sequence split** — used when no groups exist (all positives from
///    one condition).  Sequences are never split across train/val, so no
///    window leaks between the two sets.
/// 3. **No validation** — when neither split can produce validation data, the
///    validation set is empty and the caller skips early stopping and
///    threshold calibration.  The pre-fix per-window fallback is deliberately
///    gone: it re-introduced exactly the blind validation this ticket
///    eliminates.
///
/// After the split, the caller can continue to use `rng` for subsequent
/// operations (e.g. epoch-level shuffling in the Conv1D training loop).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::type_complexity
)]
fn split_train_val(
    seq_infos: &[SeqInfo],
    windows: &[Vec<f32>],
    window_labels: &[f32],
    window_weights: &[f32],
    rng: &mut StdRng,
    stratify_by_source: bool,
) -> (
    Vec<Vec<f32>>,
    Vec<f32>,
    Vec<f32>,
    Vec<Vec<f32>>,
    Vec<f32>,
    Vec<f32>,
    SplitKind,
) {
    let pos_seq_idx: Vec<usize> = seq_infos
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_positive)
        .map(|(i, _)| i)
        .collect();
    let (neg_tr_idx, neg_val_idx) = split_negative_sequences(seq_infos, rng, stratify_by_source);

    // ── 1. Positive provenance-group holdout (mahbot-1008 Fix 1) ──
    if let Some((group, val_pos_idx)) = pick_positive_holdout_group(seq_infos) {
        let val_pos_set: std::collections::HashSet<usize> = val_pos_idx.iter().copied().collect();
        let tr_pos_idx: Vec<usize> = pos_seq_idx
            .iter()
            .copied()
            .filter(|i| !val_pos_set.contains(i))
            .collect();
        let tr_seq_idx: Vec<usize> = tr_pos_idx
            .iter()
            .copied()
            .chain(neg_tr_idx.iter().copied())
            .collect();
        let val_seq_idx: Vec<usize> = val_pos_idx
            .iter()
            .copied()
            .chain(neg_val_idx.iter().copied())
            .collect();

        let tr = VoiceVerifier::gather_windows(
            windows,
            window_labels,
            window_weights,
            &tr_seq_idx,
            seq_infos,
        );
        let val = VoiceVerifier::gather_windows(
            windows,
            window_labels,
            window_weights,
            &val_seq_idx,
            seq_infos,
        );
        let n_tr_pos = tr.1.iter().filter(|&&l| l > 0.5).count();
        let n_tr_neg = tr.1.len() - n_tr_pos;
        let n_val_pos = val.1.iter().filter(|&&l| l > 0.5).count();
        let n_val_neg = val.1.len() - n_val_pos;
        info!(
            "Verifier split: group-holdout ('{}') train={n_tr_pos}+{n_tr_neg} \
             val={n_val_pos}+{n_val_neg} windows",
            group.label(),
        );
        return (
            tr.0,
            tr.1,
            tr.2,
            val.0,
            val.1,
            val.2,
            SplitKind::GroupHoldout,
        );
    }

    // ── 2. Per-sequence split ──
    let n_val_pos = ((pos_seq_idx.len() as f32) * VALIDATION_SPLIT).round() as usize;
    let mut pos_shuffled = pos_seq_idx;
    pos_shuffled.shuffle(rng);
    let n_val_pos = n_val_pos.min(pos_shuffled.len());
    let val_pos_idx: Vec<usize> = pos_shuffled[..n_val_pos].to_vec();
    let tr_pos_idx: Vec<usize> = pos_shuffled[n_val_pos..].to_vec();

    let tr_seq_idx: Vec<usize> = tr_pos_idx
        .iter()
        .copied()
        .chain(neg_tr_idx.iter().copied())
        .collect();
    let val_seq_idx: Vec<usize> = val_pos_idx
        .iter()
        .copied()
        .chain(neg_val_idx.iter().copied())
        .collect();

    let tr = VoiceVerifier::gather_windows(
        windows,
        window_labels,
        window_weights,
        &tr_seq_idx,
        seq_infos,
    );
    let val = VoiceVerifier::gather_windows(
        windows,
        window_labels,
        window_weights,
        &val_seq_idx,
        seq_infos,
    );

    // ── 3. No validation data (deliberately no leaky per-window fallback) ──
    // Two triggers: (a) the per-sequence split produced an empty validation
    // set, or (b) the split would leave fewer than MIN_POSITIVE_WINDOWS
    // positive windows in training (mahbot-1011) — prepare_training_data's
    // guard checks the TOTAL before splitting, but the split can carve
    // training below the minimum, recreating the under-powered
    // positive-training regime the guard exists to prevent.  In both cases
    // ALL sequences go to training with empty validation — never a leaky
    // per-window fallback.
    let n_tr_pos = tr.1.iter().filter(|&&l| l > 0.5).count();
    if val.0.is_empty() || n_tr_pos < MIN_POSITIVE_WINDOWS {
        let reason = if val.0.is_empty() {
            "the per-sequence split produced an empty validation set"
        } else {
            "the per-sequence split would leave fewer than the \
             MIN_POSITIVE_WINDOWS positive training windows"
        };
        warn!(
            "Verifier split: {reason} — training on all sequences with no \
             early stopping or threshold calibration.  The leaky per-window \
             fallback was removed in mahbot-1008.",
        );
        // All sequences go to training; empty validation.
        let all_idx: Vec<usize> = (0..seq_infos.len()).collect();
        let all = VoiceVerifier::gather_windows(
            windows,
            window_labels,
            window_weights,
            &all_idx,
            seq_infos,
        );
        return (
            all.0,
            all.1,
            all.2,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            SplitKind::None,
        );
    }

    (
        tr.0,
        tr.1,
        tr.2,
        val.0,
        val.1,
        val.2,
        SplitKind::PerSequence,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Window helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Form windows from a per-frame embedding list for Conv1D training. (mahbot-995).
///
/// Always uses concatenated 288-dim windows (no mean-pooling, preserving
/// temporal structure).
///
/// Input can be either per-frame 96-dim embeddings (which get windowed via
/// [`form_stride1_concatenated_windows`]) or pre-windowed 288-dim data
/// (which is L2-normalized and used directly).
fn form_conv1d_sequence_windows(embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if embeddings.is_empty() {
        return Vec::new();
    }
    if embeddings[0].len() == EMBEDDING_DIM {
        // Per-frame: form stride-1 concatenated 288-dim windows.
        form_stride1_concatenated_windows(embeddings)
    } else {
        // Pre-windowed: L2-normalize and use directly.
        embeddings
            .iter()
            .map(|f| {
                let norm = f.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
                f.iter().map(|v| v / norm).collect()
            })
            .collect()
    }
}

/// Fill a mutable output slice with `VERIFIER_WINDOW_SIZE` consecutive
/// `EMBEDDING_DIM`-length embeddings from `buffer[start..start+VERIFIER_WINDOW_SIZE]`.
/// The output slice must be exactly `VERIFIER_INPUT_DIM` (= 288) elements long.
/// This is the single canonical implementation of the 3-frame concatenation
/// pattern — both `form_stride1_windows` and `voice::score_single_embedding`
/// use it, ensuring the window format stays synchronized across modules.
#[inline]
pub(crate) fn fill_verifier_window(buffer: &[Vec<f32>], start: usize, out: &mut [f32]) {
    assert_eq!(
        out.len(),
        VERIFIER_INPUT_DIM,
        "fill_verifier_window: output slice must be {VERIFIER_INPUT_DIM} elements, got {}",
        out.len(),
    );
    for j in 0..VERIFIER_WINDOW_SIZE {
        let src = &buffer[start + j];
        let dst = &mut out[j * EMBEDDING_DIM..(j + 1) * EMBEDDING_DIM];
        dst.copy_from_slice(src);
    }
}

/// Mean-pool a 288-dim concatenated 3-frame window into a 96-dim pooled vector (testing only).
///
/// Writes into a stack-allocated `[f32; EMBEDDING_DIM]` buffer to avoid heap
/// allocation on the streaming inference hot path (mahbot-874).
///
/// # Panics
///
/// Panics if `window.len() != VERIFIER_INPUT_DIM` or `out.len() != EMBEDDING_DIM`.
#[cfg(test)]
#[inline]
#[allow(clippy::cast_precision_loss)]
pub(crate) fn mean_pool_window_into(window: &[f32], out: &mut [f32]) {
    assert_eq!(
        window.len(),
        VERIFIER_INPUT_DIM,
        "mean_pool_window_into: window must be {VERIFIER_INPUT_DIM} elements, got {}",
        window.len(),
    );
    assert_eq!(
        out.len(),
        EMBEDDING_DIM,
        "mean_pool_window_into: output buffer must be {EMBEDDING_DIM} elements, got {}",
        out.len(),
    );
    let f0 = &window[0..EMBEDDING_DIM];
    let f1 = &window[EMBEDDING_DIM..2 * EMBEDDING_DIM];
    let f2 = &window[2 * EMBEDDING_DIM..3 * EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        out[i] = (f0[i] + f1[i] + f2[i]) / VERIFIER_WINDOW_SIZE as f32;
    }
}

/// Shared stride-1 window iteration primitive.
///
/// Extracts the common outer-loop scaffolding: bounds check, capacity
/// calculation, stride-1 iteration, L2-normalization, and push.  The caller
/// provides a `form_window` closure that fills a pre-allocated
/// `window_size`-element buffer for each window index `i`.
///
/// Returns empty vec if fewer than [`VERIFIER_WINDOW_SIZE`] embeddings are available.
fn stride1_windows_impl(
    embeddings: &[Vec<f32>],
    window_size: usize,
    mut form_window: impl FnMut(usize, &mut [f32]),
) -> Vec<Vec<f32>> {
    if embeddings.len() < VERIFIER_WINDOW_SIZE {
        return Vec::new();
    }
    let n = embeddings.len() - VERIFIER_WINDOW_SIZE + 1;
    let mut windows = Vec::with_capacity(n);
    for i in 0..n {
        let mut window = vec![0.0f32; window_size];
        form_window(i, &mut window);
        // L2-normalize (matching classifier convention, mahbot-870).
        let norm = window.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
        for v in &mut window {
            *v /= norm;
        }
        windows.push(window);
    }
    windows
}

/// Form stride-1 **concatenated** windows from a flat list of 96-dim embeddings.
///
/// Each window is 3 consecutive embeddings concatenated into a 288-dim vector,
/// then L2-normalized.  Consecutive windows overlap by 2 embeddings (stride 1).
/// This is the Conv1D verifier windowing function: it preserves the full
/// 288-dim temporal structure for the Conv1D layers (mahbot-995).
///
/// Uses [`fill_verifier_window`] for the concatenation (shared canonical
/// implementation with the inference hot-path in `voice.rs`).
///
/// Returns empty vec if fewer than 3 embeddings are available.
#[allow(clippy::cast_precision_loss)]
fn form_stride1_concatenated_windows(embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
    stride1_windows_impl(embeddings, VERIFIER_INPUT_DIM, |i, out| {
        fill_verifier_window(embeddings, i, out);
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Math helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Standard sigmoid function: `1 / (1 + e^{-x})` (testing only).
#[cfg(test)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Verifier-local LeakyReLU activation with slope [`LEAKY_RELU_SLOPE`]
/// (mahbot-1008 Fix 3).
///
/// Deliberately NOT the classifier's `relu()`: ReLU's dead zone was the root
/// cause of the constant-floor collapse — when every Conv1D pre-activation is
/// ≤ 0 the pooled features are `[0, 0]` and the logit degenerates to
/// `fc_bias`, an input-independent constant.  LeakyReLU keeps a small
/// (input-dependent) signal in the negative half-plane so the feature path can
/// never die completely.
#[inline]
fn leaky_relu_verifier(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = if *v > 0.0 { *v } else { LEAKY_RELU_SLOPE * *v };
    }
}

/// Conv1D inference path for the verifier (mahbot-995).
///
/// Architecture: Conv1D(96→2, k=3, padding=1) → LeakyReLU → AdaptiveAvgPool1d → Linear(2→1) → Sigmoid.
///
/// Input must be 288-dim (3 concatenated 96-dim embeddings). Panics otherwise.
///
/// Pipeline: 288-dim input → L2-normalize → Reshape to channels-first [96 × 3]
/// → Conv1D → LeakyReLU → AdaptiveAvgPool1d → Linear → Sigmoid.
///
/// # Panics
///
/// Panics if `embedding.len() != VERIFIER_INPUT_DIM`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
fn predict_conv1d(
    embedding: &[f32],
    conv_weight: &[f32],
    conv_bias: &[f32],
    fc_weight: &[f32],
    fc_bias: &[f32],
) -> f32 {
    assert_eq!(
        embedding.len(),
        VERIFIER_INPUT_DIM,
        "Conv1D verifier expects {VERIFIER_INPUT_DIM}-dim input, got {}",
        embedding.len(),
    );

    // Step 1: L2-normalize the 288-dim input (matching training pipeline).
    // Uses a stack-allocated [f32; 288] buffer to avoid heap allocation on the
    // streaming inference hot path.
    let norm: f32 = embedding
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt()
        .max(1e-10);
    let mut x = [0.0f32; VERIFIER_INPUT_DIM];
    for (i, &v) in embedding.iter().enumerate() {
        x[i] = v / norm;
    }

    // Step 2: Convert to channels-first layout [96 channels × 3 time steps].
    // Stack-buffered (mahbot-1029 D5) — the streaming hot path calls this up
    // to `1 + ensemble_members.len()` times per embedding.
    let mut cf = [0.0f32; EMBEDDING_DIM * VERIFIER_WINDOW_SIZE];
    for t in 0..VERIFIER_WINDOW_SIZE {
        for c in 0..EMBEDDING_DIM {
            cf[c * VERIFIER_WINDOW_SIZE + t] = x[t * EMBEDDING_DIM + c];
        }
    }

    // Step 3: Conv1D(96 → 2, k=3, padding=1).  Stack-buffered — same index
    // math as `wake_word_classifier::conv1d` with the fixed verifier dims.
    let mut conv_out = [0.0f32; CONV_VERIFIER_OUT * VERIFIER_WINDOW_SIZE];
    let padding = CONV_VERIFIER_KERNEL_SIZE / 2;
    for co in 0..CONV_VERIFIER_OUT {
        for li in 0..VERIFIER_WINDOW_SIZE {
            let mut s = conv_bias[co];
            for ci in 0..EMBEDDING_DIM {
                for k in 0..CONV_VERIFIER_KERNEL_SIZE {
                    let ii = li as isize + k as isize - padding as isize;
                    if ii >= 0 && ii < VERIFIER_WINDOW_SIZE as isize {
                        s += cf[ci * VERIFIER_WINDOW_SIZE + ii as usize]
                            * conv_weight
                                [(co * EMBEDDING_DIM + ci) * CONV_VERIFIER_KERNEL_SIZE + k];
                    }
                }
            }
            conv_out[co * VERIFIER_WINDOW_SIZE + li] = s;
        }
    }

    // Step 4: LeakyReLU activation (mahbot-1008 Fix 3 — replaces ReLU to
    // eliminate the dead-zone constant floor).
    leaky_relu_verifier(&mut conv_out);

    // Step 5: AdaptiveAvgPool1d (3 → 1) — average over the time dimension.
    let mut pooled = [0.0f32; CONV_VERIFIER_OUT];
    for ci in 0..CONV_VERIFIER_OUT {
        let mut s = 0.0;
        for li in 0..VERIFIER_WINDOW_SIZE {
            s += conv_out[ci * VERIFIER_WINDOW_SIZE + li];
        }
        pooled[ci] = s / VERIFIER_WINDOW_SIZE as f32;
    }

    // Step 6: Linear(2 → 1) → Sigmoid.
    let logit: f32 = pooled
        .iter()
        .zip(fc_weight.iter())
        .map(|(v, w)| v * w)
        .sum::<f32>()
        + fc_bias[0];
    1.0 / (1.0 + (-logit).exp())
}

/// Compute weighted binary cross-entropy loss for Conv1D verifier (mahbot-995).
///
/// Runs a forward pass through the Conv1D architecture for each sample.
/// When `use_l2` is true, includes L2 regularization on conv_weight, fc_weight,
/// AND both bias terms (bias L2 added in mahbot-1008 Fix 3 — the unregularized
/// `fc_bias` was free to drift to −16.52 under ~85% negative-only batches).
#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
fn compute_conv1d_loss(
    windows: &[Vec<f32>],
    labels: &[f32],
    sample_weights: &[f32],
    conv_weight: &[f32],
    conv_bias: &[f32],
    fc_weight: &[f32],
    fc_bias: &[f32],
    cin: usize,
    cout: usize,
    ks: usize,
    lin: usize,
    l2_lambda: f32,
) -> f32 {
    let n = windows.len();
    if n == 0 {
        return 0.0;
    }
    let mut total = 0.0f32;
    for i in 0..n {
        let x = &windows[i];
        let cf = crate::audio::wake_word_classifier::to_channels_first(x, cin, lin);
        let conv_out = crate::audio::wake_word_classifier::conv1d(
            &cf,
            cin,
            lin,
            cout,
            ks,
            conv_weight,
            conv_bias,
        );
        let mut relu_out = conv_out;
        leaky_relu_verifier(&mut relu_out);
        let pooled = crate::audio::wake_word_classifier::adaptive_avg_pool(&relu_out, cout, lin);
        let logit: f32 = pooled
            .iter()
            .zip(fc_weight.iter())
            .map(|(v, w)| v * w)
            .sum::<f32>()
            + fc_bias[0];
        let pred = 1.0 / (1.0 + (-logit).exp());
        let eps = 1e-10;
        total += sample_weights[i]
            * (labels[i] * (pred + eps).ln() + (1.0 - labels[i]) * (1.0 - pred + eps).ln());
    }
    let bce = -total / n as f32;
    if l2_lambda > 0.0 {
        let l2_term = 0.5
            * l2_lambda
            * (conv_weight.iter().map(|w| w * w).sum::<f32>()
                + conv_bias.iter().map(|w| w * w).sum::<f32>()
                + fc_weight.iter().map(|w| w * w).sum::<f32>()
                + fc_bias.iter().map(|w| w * w).sum::<f32>());
        bce + l2_term
    } else {
        bce
    }
}

/// Log verifier training diagnostics and check for discrimination collapse.
#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
fn log_verifier_diagnostics(
    verifier: &VoiceVerifier,
    tr_windows: &[Vec<f32>],
    tr_labels: &[f32],
    val_windows: &[Vec<f32>],
    val_labels: &[f32],
    split_kind: SplitKind,
    class_weight: f32,
    label: &str,
) {
    let n_tr_pos = tr_labels.iter().filter(|&&l| l > 0.5).count();
    let n_tr_neg = tr_labels.len() - n_tr_pos;
    let n_val_pos = val_labels.iter().filter(|&&l| l > 0.5).count();
    let n_val_neg = val_labels.len().saturating_sub(n_val_pos);

    let mut pos_scores_tr = Vec::new();
    let mut neg_scores_tr = Vec::new();
    for (emb, &lbl) in tr_windows.iter().zip(tr_labels.iter()) {
        let score = verifier.predict(emb);
        if lbl > 0.5 {
            pos_scores_tr.push(score);
        } else {
            neg_scores_tr.push(score);
        }
    }
    let pos_mean_tr = if pos_scores_tr.is_empty() {
        0.0
    } else {
        pos_scores_tr.iter().sum::<f32>() / pos_scores_tr.len() as f32
    };
    let neg_mean_tr = if neg_scores_tr.is_empty() {
        0.0
    } else {
        neg_scores_tr.iter().sum::<f32>() / neg_scores_tr.len() as f32
    };

    let mut pos_scores_val = Vec::new();
    let mut neg_scores_val = Vec::new();
    for (emb, &lbl) in val_windows.iter().zip(val_labels.iter()) {
        let score = verifier.predict(emb);
        if lbl > 0.5 {
            pos_scores_val.push(score);
        } else {
            neg_scores_val.push(score);
        }
    }
    let pos_mean_val = if pos_scores_val.is_empty() {
        0.0
    } else {
        pos_scores_val.iter().sum::<f32>() / pos_scores_val.len() as f32
    };
    let neg_mean_val = if neg_scores_val.is_empty() {
        0.0
    } else {
        neg_scores_val.iter().sum::<f32>() / neg_scores_val.len() as f32
    };

    let split_method = match split_kind {
        SplitKind::GroupHoldout => "group-holdout (out-of-session)",
        SplitKind::PerSequence => "per-sequence",
        SplitKind::None => "no-validation",
    };
    info!(
        "{label} training ({split_method} split): \
         train={n_tr_pos}+{n_tr_neg} val={n_val_pos}+{n_val_neg} windows, \
         class_weight={class_weight:.2} | \
         train pos: mean={pos_mean_tr:.4} neg: mean={neg_mean_tr:.4} | \
         val pos: mean={pos_mean_val:.4} neg: mean={neg_mean_val:.4}"
    );

    if !pos_scores_val.is_empty() && !neg_scores_val.is_empty() {
        let ratio = pos_mean_val / neg_mean_val.max(1e-10);
        if ratio < 1.1 {
            warn!(
                "{label} discrimination low: mean_pos_val/mean_neg_val={ratio:.4} (< 1.1). \
                 The verifier may have collapsed to a near-constant predictor."
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Threshold auto-calibration (mahbot-997)
// ═══════════════════════════════════════════════════════════════════════════

/// λ value for Constrained Weighted Youden's J during threshold calibration.
///
/// Weighting false accepts (FPR) twice as heavily as false rejects (1-TPR):
/// maximize TPR - λ×FPR subject to TPR ≥ 0.90.
/// See mahbot-997 for the calibration protocol.
pub(crate) const CALIBRATION_LAMBDA: f32 = 2.0;

/// Minimum number of positive or negative validation samples required for
/// threshold calibration.  Below this threshold, fall back to the caller-supplied
/// value and emit a warning.
const CALIBRATION_MIN_SAMPLES: usize = 5;

/// Step size for threshold sweep during auto-calibration.
///
/// 0.01 → 101 candidates in [0.0, 1.0], balancing precision (~0.01 resolution)
/// with computational cost (trivial for small validation sets).
pub(crate) const CALIBRATION_SWEEP_STEP: f32 = 0.01;

/// Auto-calibrate the verifier decision threshold using Constrained Weighted
/// Youden's J on held-out validation scores.
///
/// Sweeps candidate thresholds from 0.0 to 1.0 in [`CALIBRATION_SWEEP_STEP`]
/// increments, computing:
///
/// ```text
/// Youden(T) = TPR(T) - λ × FPR(T)
/// ```
///
/// subject to `TPR(T) ≥ 0.90`.  The threshold with the maximum Youden index
/// is selected.  λ = [`CALIBRATION_LAMBDA`] (2.0) weights false accepts twice
/// as heavily as false rejects.
///
/// # Fallback
///
/// If either `pos_scores` or `neg_scores` has fewer than
/// [`CALIBRATION_MIN_SAMPLES`] (5) entries, emits a `warn!` and returns
/// `default_threshold` unchanged.
///
/// # Returns
///
/// The calibrated threshold in `(0.0, 1.0]`, or `default_threshold` if the
/// validation set is too sparse for meaningful calibration.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn calibrate_verifier_threshold(
    pos_scores: &[f32],
    neg_scores: &[f32],
    default_threshold: f32,
) -> f32 {
    if pos_scores.len() < CALIBRATION_MIN_SAMPLES || neg_scores.len() < CALIBRATION_MIN_SAMPLES {
        warn!(
            "Verifier threshold calibration skipped: pos={}, neg={} validation samples \
             (need ≥{} each).  Using caller-supplied threshold={:.4}.",
            pos_scores.len(),
            neg_scores.len(),
            CALIBRATION_MIN_SAMPLES,
            default_threshold,
        );
        return default_threshold;
    }

    let n_pos = pos_scores.len() as f32;
    let n_neg = neg_scores.len() as f32;

    // Sweep thresholds from 0.0 to 1.0 in CALIBRATION_SWEEP_STEP increments.
    let n_steps = (1.0 / CALIBRATION_SWEEP_STEP).round() as usize + 1;
    let mut best_youden = f32::NEG_INFINITY;
    let mut best_threshold = default_threshold;

    for step in 0..n_steps {
        let t = (step as f32) * CALIBRATION_SWEEP_STEP;

        // TPR: fraction of positives with score >= threshold.
        let tp = pos_scores.iter().filter(|&&s| s >= t).count() as f32;
        let tpr = tp / n_pos;

        // FPR: fraction of negatives with score >= threshold.
        let fp = neg_scores.iter().filter(|&&s| s >= t).count() as f32;
        let fpr = fp / n_neg;

        // Constrained Weighted Youden's J.
        // Use `>=` to prefer higher thresholds for the same Youden value
        // (higher threshold = fewer false accepts).
        // Note: at t=0.0, TPR is always 1.0 (sigmoid outputs are in [0, 1]),
        // so at least one threshold always satisfies TPR ≥ 0.90.
        if tpr >= 0.90 {
            let youden = tpr - CALIBRATION_LAMBDA * fpr;
            if youden >= best_youden {
                best_youden = youden;
                best_threshold = t;
            }
        }
    }

    info!(
        "Verifier threshold calibration: selected threshold={best_threshold:.4} \
         (Youden={best_youden:.4}, λ={CALIBRATION_LAMBDA:.1}), \
         pos={pos} neg={neg} validation samples",
        pos = pos_scores.len(),
        neg = neg_scores.len(),
    );
    best_threshold
}

// ═══════════════════════════════════════════════════════════════════════════
// Synthetic negatives
// ═══════════════════════════════════════════════════════════════════════════

/// Generate synthetic negative embeddings based on the statistics of the
/// positive embeddings (mahbot-846).  Unlike pure N(0,1) Gaussian noise
/// which sits in a completely different region of embedding space than real
/// speech, this function produces negatives that overlap with the real
/// embedding distribution.
///
/// Each synthetic negative is sampled as:
///   `mean + noise_scale * sigma * N(0, 1)`
/// per dimension, then L2-normalised to the unit sphere.  This puts the
/// synthetic negatives in the same region of embedding space as the real
/// positives, providing useful training signal for the wake word vs.
/// confusable boundary.
///
/// `noise_scale` controls how far the negatives are pushed from the positive
/// centroid (default 1.5 — large enough to create a separation margin while
/// maintaining distribution overlap).
///
/// When `rng_seed` is `Some(seed)`, a seeded `StdRng` is used for all random
/// operations, making generation deterministic.  When `None`, entropy-based
/// randomness is used (production path).
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub(crate) fn generate_synthetic_negatives_from_positives(
    count: usize,
    positives: &[Vec<f32>],
    noise_scale: f32,
    rng_seed: Option<u64>,
) -> Vec<Vec<f32>> {
    if positives.is_empty() || count == 0 {
        return vec![];
    }
    let dim = positives[0].len();

    // Compute per-dimension mean and std of positive embeddings.
    let mut mean = vec![0.0; dim];
    for emb in positives {
        for (m, &v) in mean.iter_mut().zip(emb.iter()) {
            *m += v;
        }
    }
    for m in &mut mean {
        *m /= positives.len() as f32;
    }

    let mut std = vec![0.0; dim];
    for emb in positives {
        for ((s, &v), &m) in std.iter_mut().zip(emb.iter()).zip(mean.iter()) {
            *s += (v - m) * (v - m);
        }
    }
    let n = positives.len() as f32;
    for s in &mut std {
        *s = (*s / n).sqrt().max(1e-6);
    }

    // Create RNG: seeded for determinism or entropy-based for production.
    let mut rng: StdRng = if let Some(seed) = rng_seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::seed_from_u64(rand::random())
    };

    (0..count)
        .map(|_| {
            // Pick a random positive as the base (adds diversity).
            let base = &positives[rng.random_range(0..positives.len())];
            let mut emb: Vec<f32> = base
                .iter()
                .zip(std.iter())
                .map(|(&b, &s)| {
                    // Box-Muller N(0,1) — shared helper (mahbot-1043); the
                    // sin branch is discarded exactly as the inline copy did.
                    let (z, _) = crate::util::sample_gaussian_pair(&mut rng);
                    // Perturb the base embedding: move away by noise_scale * sigma
                    // This puts the synthetic negative in the same region as real
                    // speech but shifted toward the distribution tails.
                    b + noise_scale * s * z
                })
                .collect();

            // L2-normalize to unit sphere (matching real embeddings).
            let norm = emb.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
            for v in &mut emb {
                *v /= norm;
            }
            emb
        })
        .collect()
}

/// Build the synthetic-negative [`EmbeddingSequence`] used by synthetic
/// negative training: distribution-matched negatives generated from the flat
/// positive embeddings (mahbot-797), wrapped as a single synthetic negative
/// sequence.
///
/// When `rng_seed` is `Some(seed)`, generation is deterministic.  The
/// multi-seed ensemble ([`VoiceVerifier::train_ensemble_with_synthetic_negatives`])
/// generates the sequence ONCE from its entropy base seed and shares it
/// across all members, so member diversity comes purely from the training
/// seed (identical-data seed ensemble, mahbot-1025).
fn build_synthetic_negative_sequence(
    flat_positives: &[Vec<f32>],
    rng_seed: Option<u64>,
) -> EmbeddingSequence {
    let negatives = generate_synthetic_negatives_from_positives(
        SYNTHETIC_NEGATIVES_COUNT,
        flat_positives,
        1.5, // noise_scale — matched to benchmark default
        rng_seed,
    );
    EmbeddingSequence::negative(
        crate::audio::embedding_sequence::UtteranceId {
            sequence_index: 0,
            variant_index: 0,
        },
        crate::audio::embedding_sequence::Source::Synthetic,
        None,
        negatives,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Embedding pooling
// ═══════════════════════════════════════════════════════════════════════════

/// Mean-pool a sequence of per-frame embeddings (from one utterance) into a
/// single 96-dim embedding vector.
///
/// This is used during verifier training to convert a sequence of per-frame
/// embeddings from one enrollment utterance into a single positive example.
///
/// Returns an empty `Vec` when `embeddings` is empty.
///
/// Note: As of mahbot-788 Fix 3, the verifier training uses per-frame
/// embeddings directly instead of mean-pooled vectors. This function is
/// now used by [`validate_enrollment_consistency`](crate::audio::voice::validate_enrollment_consistency)
/// to compute per-utterance means for centroid cosine-similarity analysis.
/// It remains available for any other use that needs utterance-level pooling.
#[must_use]
pub fn mean_pool_embeddings(embeddings: &[Vec<f32>]) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }
    let dim = embeddings[0].len();
    if dim == 0 {
        return Vec::new();
    }
    #[allow(clippy::cast_precision_loss)]
    let n = embeddings.len() as f32;
    let mut mean = vec![0.0; dim];
    for emb in embeddings {
        for (i, &v) in emb.iter().enumerate() {
            mean[i] += v;
        }
    }
    for v in &mut mean {
        *v /= n;
    }
    mean
}

/// Verify that a contiguous range of negative-embedding weights all equal the
/// expected value.
///
/// This is a structural guard that detects silent misalignment between
/// negative embedding concatenation order and per-negative weight tier
/// assignment.  Each tier corresponds to a specific category of negative
/// embeddings (ambient, unrelated, confusable, synthetic, etc.) and all
/// weights in that tier should be identical.
///
/// Used by production [`finalize_enrollment`](crate::audio::voice::finalize_enrollment)
/// and both paths in the E2E benchmark to ensure weight tiers stay aligned with
/// embedding concatenation order across refactors.
///
/// # Panics
///
/// Panics if any weight in `weights[offset..offset + count]` differs from
/// `expected` by more than [`f32::EPSILON`].
#[inline]
pub(crate) fn assert_weight_tier(
    weights: &[f32],
    offset: usize,
    count: usize,
    expected: f32,
    label: &str,
) {
    for (j, &w) in weights[offset..offset + count].iter().enumerate() {
        let i = offset + j;
        assert!(
            (w - expected).abs() <= f32::EPSILON,
            "Weight tier mismatch: {label} weight at position {i} should be {expected}, got {w}",
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    // Shared test fixture (mahbot-1043): single authoritative make_seq body
    // lives next to the EmbeddingSequence type; local builders drifted once.
    use crate::audio::embedding_sequence::make_test_sequence as make_seq;

    /// Generate a synthetic 288-dim "positive" window with values clustered
    /// around +0.5 (simulating a wake-word embedding window).
    fn make_positive_embedding(rng: &mut impl Rng) -> Vec<f32> {
        (0..VERIFIER_INPUT_DIM)
            .map(|_| {
                // Positive cluster: N(0.5, 0.3).  Shared sampler (mahbot-1043)
                // preserves the pre-extraction `0.3 * r * cos(theta)` ordering.
                0.5 + crate::util::sample_gaussian_scaled(rng, 0.3)
            })
            .collect()
    }

    /// Generate a synthetic 288-dim "negative" window with values clustered
    /// around -0.5 (simulating a non-wake-word embedding window).
    fn make_negative_embedding(rng: &mut impl Rng) -> Vec<f32> {
        (0..VERIFIER_INPUT_DIM)
            .map(|_| {
                // Negative cluster: N(-0.5, 0.3)
                -0.5 + crate::util::sample_gaussian_scaled(rng, 0.3)
            })
            .collect()
    }

    // ── Required tests (from ticket mahbot-777) ─────────────────────

    /// Generate a synthetic 288-dim "non-wake-word" window with values
    /// distributed near 0 (simulating real non-wake-word speech or ambient
    /// audio that survives Conv1D matching).  Unlike the old opposite-direction
    /// negatives (N(-0.5, 0.3)), these sit in the same general region as
    /// wake word embeddings but lack the consistent structure that the
    /// verifier must learn to discriminate (mahbot-797).
    fn make_non_wake_speech_embedding(rng: &mut impl Rng) -> Vec<f32> {
        (0..VERIFIER_INPUT_DIM)
            .map(|_| {
                // Broad cluster centered at 0 with higher variance: N(0, 0.6).
                // This simulates the diversity of non-wake-word speech —
                // some dimensions may overlap with the wake word cluster,
                // making discrimination harder than the old opposite-direction
                // negatives.
                0.0 + crate::util::sample_gaussian_scaled(rng, 0.6)
            })
            .collect()
    }

    // ── assert_weight_tier tests (mahbot-880 reviewer feedback) ────────

    #[test]
    fn assert_weight_tier_all_match() {
        // Normal case: all weights match expected value
        let weights = vec![1.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        assert_weight_tier(&weights, 0, 3, 1.0, "first");
        assert_weight_tier(&weights, 3, 2, 2.0, "second");
        assert_weight_tier(&weights, 5, 1, 3.0, "third");

        // Edge case: count=0 should not panic at any offset
        let empty_weights: Vec<f32> = vec![1.0, 2.0, 3.0];
        assert_weight_tier(&empty_weights, 0, 0, 1.0, "empty-at-start");
        assert_weight_tier(&empty_weights, 1, 0, 0.0, "empty-at-middle");
        assert_weight_tier(&empty_weights, 3, 0, 0.0, "empty-at-end"); // Mismatch should panic with descriptive message — verify via catch_unwind.
        let mismatch_weights = vec![1.0, 1.0, 2.0];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_weight_tier(&mismatch_weights, 0, 3, 1.0, "first");
        }));
        assert!(
            result.is_err(),
            "assert_weight_tier should panic on mismatch"
        );
    }

    // ── Required tests (from ticket mahbot-777) ─────────────────────

    #[test]
    fn test_verifier_accepts_positive_rejects_negative() {
        // Train on known positive and negative synthetic embeddings, then verify
        // both acceptance of held-out positives and rejection of held-out negatives
        // (consolidated from two separate tests with identical setup, mahbot-874).
        let mut rng = StdRng::seed_from_u64(42);
        let positives: Vec<Vec<f32>> = (0..40).map(|_| make_positive_embedding(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..30).map(|_| make_negative_embedding(&mut rng)).collect();
        let pos_seq = make_seq(
            positives,
            crate::audio::embedding_sequence::LabelStratum::Positive,
        );
        let neg_seq = make_seq(
            negatives,
            crate::audio::embedding_sequence::LabelStratum::Negative,
        );

        let verifier = VoiceVerifier::train(
            &[pos_seq],
            &[neg_seq],
            None,                       // no per-negative weights
            DEFAULT_VERIFIER_THRESHOLD, // threshold
            0.001,                      // weak L2 (clean synthetic data)
            Some(42),                   // deterministic seed for reproducibility
        );

        assert!(verifier.is_trained(), "Verifier must be trained");

        // Verify a held-out positive is accepted.
        let held_out_pos = make_positive_embedding(&mut rng);
        let score_pos = verifier.predict(&held_out_pos);
        assert!(
            score_pos >= 0.5,
            "Verifier should accept positive embedding (score >= 0.5), got score={score_pos:.4}",
        );

        // Verify a held-out negative is rejected.
        let held_out_neg = make_negative_embedding(&mut rng);
        let score_neg = verifier.predict(&held_out_neg);
        assert!(
            score_neg < 0.5,
            "Verifier should reject negative embedding (score < 0.5), got score={score_neg:.4}",
        );

        // Structural assertions: Conv1D weights dimensions (mahbot-995).
        let expected_conv_w = CONV_VERIFIER_OUT * EMBEDDING_DIM * CONV_VERIFIER_KERNEL_SIZE;
        assert_eq!(
            verifier.conv_weight.len(),
            expected_conv_w,
            "conv_weight must be {expected_conv_w}-dim",
        );
        assert_eq!(
            verifier.conv_bias.len(),
            CONV_VERIFIER_OUT,
            "conv_bias must be {CONV_VERIFIER_OUT}-dim",
        );
        assert_eq!(
            verifier.fc_weight.len(),
            CONV_VERIFIER_OUT,
            "fc_weight must be {CONV_VERIFIER_OUT}-dim",
        );
        assert_eq!(verifier.fc_bias.len(), 1, "fc_bias must be 1-dim");
        assert!(
            verifier.conv_weight.iter().all(|v| v.is_finite()),
            "all conv_weight entries must be finite",
        );

        // Verify discrimination on held-out 288-dim windows.
        let held_out_pos_win = make_positive_embedding(&mut rng);
        let held_out_neg_win = make_negative_embedding(&mut rng);
        let score_pos = verifier.predict(&held_out_pos_win);
        let score_neg = verifier.predict(&held_out_neg_win);
        assert!(
            score_pos > score_neg,
            "Verifier must discriminate: pos={score_pos:.4} neg={score_neg:.4}",
        );
    }

    // ── Mahbot-797: real-negative tests ─────────────────────────────

    #[test]
    fn test_verifier_rejects_non_wake_speech() {
        // Train on positive embeddings (N(0.5, 0.3)) and realistic
        // non-wake-word embeddings (N(0, 0.6)) — these overlap with the
        // positive cluster, requiring the verifier to learn a more nuanced
        // boundary than the old opposite-direction test.
        let mut rng = StdRng::seed_from_u64(42);
        let positives: Vec<Vec<f32>> = (0..40).map(|_| make_positive_embedding(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..30)
            .map(|_| make_non_wake_speech_embedding(&mut rng))
            .collect();
        let pos_seq = make_seq(
            positives,
            crate::audio::embedding_sequence::LabelStratum::Positive,
        );
        let neg_seq = make_seq(
            negatives,
            crate::audio::embedding_sequence::LabelStratum::Negative,
        );

        let verifier = VoiceVerifier::train(
            &[pos_seq],
            &[neg_seq],
            None,                       // no per-negative weights
            DEFAULT_VERIFIER_THRESHOLD, // mahbot-853: lowered from 0.6 for streaming inference.
            CONV_L2_LAMBDA,             // Conv1D L2 regularization (mahbot-994)
            Some(42),                   // deterministic seed for reproducibility
        );

        assert!(verifier.is_trained(), "Verifier must be trained");

        // Verify discrimination: held-out positive > held-out negative.
        // Uses relative comparison (scores are not calibrated to absolute thresholds).
        let held_out_pos = make_positive_embedding(&mut rng);
        let held_out_neg = make_non_wake_speech_embedding(&mut rng);
        let score_pos = verifier.predict(&held_out_pos);
        let score_neg = verifier.predict(&held_out_neg);
        assert!(
            score_pos >= 0.4,
            "Verifier should score positive >= 0.4, got {score_pos:.4}",
        );
        assert!(
            score_pos > score_neg,
            "Verifier must discriminate: pos={score_pos:.4} neg={score_neg:.4}",
        );
    }

    #[test]
    fn test_synthetic_negatives_reject_non_wake_word_speech() {
        // Tests the synthetic-negative verifier training mechanism (mahbot-797):
        // when fewer than 2 real negative chunks are available, the verifier is
        // trained with distribution-matched synthetic negatives generated from
        // the positive pool.  This verifies that the resulting decision boundary
        // correctly rejects non-wake-word speech embeddings.  The single-member
        // construction below (shared helper + Self::train with a fixed seed) is
        // deterministic; the multi-seed production variant is covered by
        // test_ensemble_with_synthetic_negatives_trains_all_members.
        let mut rng = StdRng::seed_from_u64(99);
        let positives: Vec<Vec<f32>> = (0..32).map(|_| make_positive_embedding(&mut rng)).collect();
        let pos_seq = make_seq(
            positives.clone(),
            crate::audio::embedding_sequence::LabelStratum::Positive,
        );
        let flat_positives: Vec<Vec<f32>> = positives;
        let synth_seq = build_synthetic_negative_sequence(&flat_positives, Some(99));
        let verifier = VoiceVerifier::train(
            &[pos_seq],
            &[synth_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            CONV_L2_LAMBDA,
            Some(99),
        );

        assert!(verifier.is_trained(), "Verifier must be trained");
        assert_eq!(
            verifier.threshold, DEFAULT_VERIFIER_THRESHOLD,
            "threshold must match DEFAULT_VERIFIER_THRESHOLD",
        );

        // Structural assertions: Conv1D weights dimensions.
        let expected_conv_w = CONV_VERIFIER_OUT * EMBEDDING_DIM * CONV_VERIFIER_KERNEL_SIZE;
        assert_eq!(
            verifier.conv_weight.len(),
            expected_conv_w,
            "conv_weight must be {expected_conv_w}-dim",
        );
        assert_eq!(
            verifier.conv_bias.len(),
            CONV_VERIFIER_OUT,
            "conv_bias must be {CONV_VERIFIER_OUT}-dim",
        );
        assert_eq!(
            verifier.fc_weight.len(),
            CONV_VERIFIER_OUT,
            "fc_weight must be {CONV_VERIFIER_OUT}-dim",
        );
        assert_eq!(verifier.fc_bias.len(), 1, "fc_bias must be 1-dim");
        for &w in verifier
            .conv_weight
            .iter()
            .chain(verifier.conv_bias.iter())
            .chain(verifier.fc_weight.iter())
            .chain(verifier.fc_bias.iter())
        {
            assert!(w.is_finite(), "all weights must be finite");
        }

        // Verify a held-out positive is accepted.  NOTE (mahbot-1008): the
        // absolute score scale shifted slightly with LeakyReLU + bias L2/clamp;
        // the meaningful bar is a score well above the reject-all floor (the
        // pre-fix collapse produced 6.67e-8).  Discrimination is asserted below.
        let held_out = make_positive_embedding(&mut rng);
        let score = verifier.predict(&held_out);
        assert!(
            score >= 0.4,
            "Verifier should score positive >= 0.4, got score={score:.4}",
        );

        // Verify discrimination: positive score > non-wake-word score.
        let held_out_non_wake = make_non_wake_speech_embedding(&mut rng);
        let score_non_wake = verifier.predict(&held_out_non_wake);
        // Note: synthetic negatives are distribution-matched to the positives'
        // statistics. With only synthetic training data, absolute rejection
        // scores may vary — we verify the verifier produces a meaningful
        // positive score (≥ 0.4) rather than an absolute rejection threshold.
        // Real-negative discrimination is covered by
        // test_verifier_rejects_non_wake_speech.
        assert!(
            score >= 0.4,
            "Verifier should score positive >= 0.4, got {score:.4}",
        );
        assert!(score.is_finite(), "Positive score must be finite");
        assert!(score_non_wake.is_finite(), "Non-wake score must be finite");
        // The verifier must actually discriminate: held-out positives score
        // above held-out non-wake speech (mahbot-1008 — the old absolute
        // assertions could pass for a constant predictor).
        assert!(
            score > score_non_wake,
            "Verifier must discriminate: pos={score:.4} non_wake={score_non_wake:.4}",
        );
    }

    #[test]
    fn test_ensemble_with_synthetic_negatives_trains_all_members() {
        // The multi-seed synthetic-negative ensemble (production fallback path,
        // mahbot-1025) must train every member over the shared synthetic
        // negative set and satisfy the common ensemble invariants (per-member
        // validity, mean prediction, serialization roundtrip — see
        // assert_ensemble_properties).
        let mut rng = StdRng::seed_from_u64(1025);
        let positives: Vec<Vec<f32>> = (0..32).map(|_| make_positive_embedding(&mut rng)).collect();
        let pos_seq = make_seq(
            positives,
            crate::audio::embedding_sequence::LabelStratum::Positive,
        );
        let ensemble = VoiceVerifier::train_ensemble_with_synthetic_negatives(
            &[pos_seq],
            DEFAULT_VERIFIER_THRESHOLD,
        );
        let held_out = make_positive_embedding(&mut rng);
        assert_ensemble_properties(&ensemble, &held_out, "synthetic-negative ensemble");
    }

    /// Shared ensemble-property assertions (mahbot-1025), used by both
    /// ensemble tests (real-negative and synthetic-negative training paths).
    /// Asserts the common invariants: every member is individually trained and
    /// single-member, `predict()` equals the arithmetic mean of the member
    /// scores on a held-out vector, and the serde roundtrip preserves the
    /// ensemble and its predictions.  Pure structural/deterministic assertions
    /// (the training seed is entropy-drawn per run, so no absolute score
    /// thresholds are checked).
    fn assert_ensemble_properties(ensemble: &VoiceVerifier, held_out: &[f32], label: &str) {
        assert!(ensemble.is_trained(), "{label}: ensemble must be trained");
        assert_eq!(
            ensemble.ensemble_size(),
            VERIFIER_ENSEMBLE_SEEDS,
            "{label}: ensemble size must equal VERIFIER_ENSEMBLE_SEEDS",
        );
        assert_eq!(
            ensemble.ensemble_members.len(),
            VERIFIER_ENSEMBLE_SEEDS - 1,
            "{label}: members stored in ensemble_members (excluding primary)",
        );
        assert!(
            !ensemble.is_collapsed(),
            "{label}: healthy ensemble must not be collapsed",
        );

        // Every member must be individually trained and structurally valid.
        for idx in 0..ensemble.ensemble_size() {
            let mv = ensemble.member_only(idx);
            assert!(mv.is_trained(), "{label}: member {idx} must be trained");
            assert_eq!(
                mv.ensemble_size(),
                1,
                "{label}: member_only view is single-member",
            );
        }

        // predict() == mean of member predictions on a held-out vector.
        let ensemble_score = ensemble.predict(held_out);
        let member_scores: Vec<f32> = (0..ensemble.ensemble_size())
            .map(|idx| ensemble.member_only(idx).predict(held_out))
            .collect();
        let member_mean = member_scores.iter().copied().sum::<f32>() / member_scores.len() as f32;
        assert!(
            (ensemble_score - member_mean).abs() < 1e-4,
            "{label}: ensemble predict must equal member mean: ensemble={ensemble_score:.4} \
             mean={member_mean:.4} members={member_scores:?}",
        );

        // Serialization roundtrip preserves the ensemble and its predictions.
        let json = serde_json::to_string(ensemble).expect("serialize ensemble");
        let deserialized: VoiceVerifier =
            serde_json::from_str(&json).expect("deserialize ensemble");
        assert!(
            deserialized.is_trained(),
            "{label}: deserialized ensemble must be trained",
        );
        assert_eq!(
            deserialized.ensemble_size(),
            VERIFIER_ENSEMBLE_SEEDS,
            "{label}: deserialized ensemble must keep all members",
        );
        let after = deserialized.predict(held_out);
        assert!(
            (after - ensemble_score).abs() < 1e-4,
            "{label}: ensemble prediction must survive roundtrip: before={ensemble_score:.4} after={after:.4}",
        );
    }

    #[test]
    fn test_verifier_noop_when_untrained() {
        let verifier = VoiceVerifier::untrained();
        assert!(!verifier.is_trained());

        // Should accept any embedding with score 1.0 (no-op).
        let embedding = vec![0.5; VERIFIER_INPUT_DIM];
        let score = verifier.predict(&embedding);
        assert!(
            (score - 1.0).abs() < 1e-6,
            "Untrained verifier should return 1.0, got {score}",
        );
    }

    #[test]
    fn test_verifier_serialization_roundtrip() {
        // Train a Conv1D model and verify JSON roundtrip preserves predictions
        // and is_trained() status.
        let mut rng = StdRng::seed_from_u64(42);
        let positives: Vec<Vec<f32>> = (0..40).map(|_| make_positive_embedding(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..30).map(|_| make_negative_embedding(&mut rng)).collect();
        let pos_seq = make_seq(
            positives,
            crate::audio::embedding_sequence::LabelStratum::Positive,
        );
        let neg_seq = make_seq(
            negatives,
            crate::audio::embedding_sequence::LabelStratum::Negative,
        );

        let verifier = VoiceVerifier::train(
            &[pos_seq],
            &[neg_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            0.001,
            Some(42),
        );

        assert!(
            verifier.is_trained(),
            "Verifier must be trained for roundtrip test",
        );

        // Serialize to JSON.
        let json = serde_json::to_string(&verifier).expect("serialize");

        // Deserialize.
        let deserialized: VoiceVerifier = serde_json::from_str(&json).expect("deserialize");

        // Verify is_trained() works on deserialized model.
        assert!(
            deserialized.is_trained(),
            "deserialized verifier should be trained",
        );

        // Verify predictions match on held-out test vectors.
        let held_out_pos = make_positive_embedding(&mut rng);
        let held_out_neg = make_negative_embedding(&mut rng);

        let score_before = verifier.predict(&held_out_pos);
        let score_after = deserialized.predict(&held_out_pos);
        assert!(
            (score_before - score_after).abs() < 1e-4,
            "Positive prediction must match after roundtrip: before={score_before:.4} after={score_after:.4}",
        );

        let score_before = verifier.predict(&held_out_neg);
        let score_after = deserialized.predict(&held_out_neg);
        assert!(
            (score_before - score_after).abs() < 1e-4,
            "Negative prediction must match after roundtrip: before={score_before:.4} after={score_after:.4}",
        );

        // Untrained verifier serialization roundtrip must remain no-op.
        let untrained = VoiceVerifier::untrained();
        let json = serde_json::to_string(&untrained).expect("serialize");
        let deserialized_untrained: VoiceVerifier =
            serde_json::from_str(&json).expect("deserialize");
        assert!(!deserialized_untrained.is_trained());
        let score = deserialized_untrained.predict(&[0.0; VERIFIER_INPUT_DIM]);
        assert!((score - 1.0).abs() < 1e-6);
    }

    // ── Multi-seed ensemble tests (mahbot-1025) ────────────────────────

    #[test]
    fn test_ensemble_trains_all_members_and_averages() {
        // The ensemble trains VERIFIER_ENSEMBLE_SEEDS members and predict()
        // returns their MEAN score.  Member 0 becomes the primary; the rest
        // attach in ensemble_members.  The common invariants (per-member
        // validity, mean == predict, roundtrip) live in
        // assert_ensemble_properties.
        let mut rng = StdRng::seed_from_u64(1025);
        let positives: Vec<Vec<f32>> = (0..60).map(|_| make_positive_embedding(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..50).map(|_| make_negative_embedding(&mut rng)).collect();
        let pos_seq = make_seq(
            positives,
            crate::audio::embedding_sequence::LabelStratum::Positive,
        );
        let neg_seq = make_seq(
            negatives,
            crate::audio::embedding_sequence::LabelStratum::Negative,
        );

        let ensemble = VoiceVerifier::train_ensemble(
            &[pos_seq],
            &[neg_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            CONV_L2_LAMBDA,
        );
        let held_out = make_positive_embedding(&mut rng);
        assert_ensemble_properties(&ensemble, &held_out, "real-negative ensemble");
    }

    #[test]
    fn test_ensemble_parallel_matches_serial_reference() {
        // mahbot-1029 D3: the parallel member-training path must produce
        // byte-identical member weights to the serial path for the same base
        // seed — the bench's member fingerprint arrays are compared across
        // runs, so any divergence (e.g. from accidentally sharing
        // split_train_val) would break the baseline comparison.
        let mut rng = StdRng::seed_from_u64(1029);
        let positives: Vec<Vec<f32>> = (0..40).map(|_| make_positive_embedding(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..40).map(|_| make_negative_embedding(&mut rng)).collect();
        let pos_seq = make_seq(
            positives,
            crate::audio::embedding_sequence::LabelStratum::Positive,
        );
        let neg_seq = make_seq(
            negatives,
            crate::audio::embedding_sequence::LabelStratum::Negative,
        );
        let base_seed: u64 = 0xdead_beef;

        // Parallel ensemble path (seeded variant, same code the public
        // wrapper runs with an entropy-drawn base seed).
        let (par, _par_metrics) = VoiceVerifier::train_ensemble_with_metrics_seeded(
            &[pos_seq.clone()],
            &[neg_seq.clone()],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            CONV_L2_LAMBDA,
            base_seed,
        );

        // Serial reference: one train_with_metrics per member with the same
        // derived seeds, assembled into an identical primary+members shape.
        let mut members: Vec<VoiceVerifier> = Vec::with_capacity(VERIFIER_ENSEMBLE_SEEDS);
        for i in 0..VERIFIER_ENSEMBLE_SEEDS {
            let seed = base_seed.wrapping_add(i as u64);
            let (member, _metrics) = VoiceVerifier::train_with_metrics(
                &[pos_seq.clone()],
                &[neg_seq.clone()],
                None,
                DEFAULT_VERIFIER_THRESHOLD,
                CONV_L2_LAMBDA,
                Some(seed),
            );
            members.push(member);
        }
        let mut serial_primary = members.remove(0);
        serial_primary.ensemble_members = members;

        assert_eq!(
            par.ensemble_size(),
            serial_primary.ensemble_size(),
            "both paths must train the same number of members"
        );
        assert!(
            par.is_trained() && serial_primary.is_trained(),
            "test data must be large enough to train both paths"
        );
        for idx in 0..par.ensemble_size() {
            let p = par.member_only(idx);
            let s = serial_primary.member_only(idx);
            assert_eq!(
                p.conv_weight, s.conv_weight,
                "member {idx} conv_weight diverges from the serial path"
            );
            assert_eq!(
                p.conv_bias, s.conv_bias,
                "member {idx} conv_bias diverges from the serial path"
            );
            assert_eq!(
                p.fc_weight, s.fc_weight,
                "member {idx} fc_weight diverges from the serial path"
            );
            assert_eq!(
                p.fc_bias, s.fc_bias,
                "member {idx} fc_bias diverges from the serial path"
            );
        }
    }

    #[test]
    fn test_ensemble_serializes_backward_compatibly() {
        // Legacy JSON (no ensemble_members field) must load as a single-member
        // verifier — pre-mahbot-1025 persisted models keep working.
        let mut rng = StdRng::seed_from_u64(7);
        let positives: Vec<Vec<f32>> = (0..40).map(|_| make_positive_embedding(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..30).map(|_| make_negative_embedding(&mut rng)).collect();
        let pos_seq = make_seq(
            positives,
            crate::audio::embedding_sequence::LabelStratum::Positive,
        );
        let neg_seq = make_seq(
            negatives,
            crate::audio::embedding_sequence::LabelStratum::Negative,
        );
        let single = VoiceVerifier::train(
            &[pos_seq],
            &[neg_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            CONV_L2_LAMBDA,
            Some(42),
        );
        // Single-member serialization must NOT emit the ensemble field
        // (skip_serializing_if empty keeps legacy bytes identical).
        let json = serde_json::to_string(&single).expect("serialize");
        assert!(
            !json.contains("ensemble_members"),
            "single-member verifier must not serialize ensemble_members: {json}",
        );
        // And a legacy-shaped JSON without the field must load.
        let legacy: VoiceVerifier = serde_json::from_str(&json).expect("deserialize legacy");
        assert!(legacy.is_trained());
        assert_eq!(legacy.ensemble_size(), 1, "legacy loads as single member");
        // A member-only view of a single-member verifier is itself.
        let member = single.member_only(0);
        assert!(member.is_trained());
        assert_eq!(member.ensemble_size(), 1);
    }

    /// Build a trained single-member verifier that rejects every input with a
    /// constant ~6.67e-8 score (the mahbot-1008 collapse brick).
    fn constant_reject_brick() -> VoiceVerifier {
        VoiceVerifier {
            trained: true,
            threshold: DEFAULT_VERIFIER_THRESHOLD,
            activation: VerifierActivation::LeakyReLU,
            conv_weight: vec![0.0; CONV_VERIFIER_OUT * EMBEDDING_DIM * CONV_VERIFIER_KERNEL_SIZE],
            conv_bias: vec![0.0; CONV_VERIFIER_OUT],
            fc_weight: vec![0.0; CONV_VERIFIER_OUT],
            fc_bias: vec![-16.52], // sigmoid ≈ 6.67e-8
            ensemble_members: Vec::new(),
        }
    }

    #[test]
    fn test_ensemble_any_collapsed_member_flags_whole() {
        // is_collapsed() must probe EVERY trained member: a single collapsed
        // (constant input-independent reject) member drags the ensemble mean
        // down on every input and CAN push a genuine wake word below the 0.86
        // floor — but the mean alone is not a reliable detector (ten healthy
        // members at ~0.97 + one brick ≈ 0.87, still above the floor), so the
        // guard is the per-member probe, not the mean crossing.  All assertions
        // here are deterministic: the flag flips once any member is the brick,
        // and the mean strictly decreases (the brick's 6.67e-8 replaces the
        // replaced member's > 1e-3 score).
        let mut rng = StdRng::seed_from_u64(1025);
        let positives: Vec<Vec<f32>> = (0..60).map(|_| make_positive_embedding(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..50).map(|_| make_negative_embedding(&mut rng)).collect();
        let pos_seq = make_seq(
            positives,
            crate::audio::embedding_sequence::LabelStratum::Positive,
        );
        let neg_seq = make_seq(
            negatives,
            crate::audio::embedding_sequence::LabelStratum::Negative,
        );
        let mut ensemble = VoiceVerifier::train_ensemble(
            &[pos_seq],
            &[neg_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            CONV_L2_LAMBDA,
        );
        assert!(!ensemble.is_collapsed(), "healthy ensemble not collapsed");

        // Fixed genuine-wake-word probe; capture the healthy ensemble score and
        // the replaced member's standalone score BEFORE poisoning.
        let probe = make_positive_embedding(&mut rng);
        let healthy_score = ensemble.predict(&probe);
        let last_idx = ensemble.ensemble_size() - 1;
        let replaced_member_score = ensemble.member_only(last_idx).predict(&probe);
        assert!(
            replaced_member_score > 1e-3,
            "test precondition: replaced member must score as a genuine wake word \
             (got {replaced_member_score:.4}); a sub-brick score would make the \
             drag assertion vacuous",
        );

        // Replace the last member with a constant reject-all brick.
        let collapsed = constant_reject_brick();
        *ensemble.ensemble_members.last_mut().expect("members exist") = collapsed;
        assert!(
            ensemble.is_collapsed(),
            "any collapsed member must flag the whole ensemble",
        );

        // The brick (≈ 6.67e-8 on every input) strictly lowers the ensemble
        // mean on the same probe — deterministic given the precondition above:
        // poisoned − healthy = (6.67e-8 − replaced_member_score) / N < 0.
        let poisoned_score = ensemble.predict(&probe);
        assert!(
            poisoned_score < healthy_score,
            "collapsed member must drag the ensemble mean down: \
             healthy={healthy_score:.4} poisoned={poisoned_score:.4}",
        );
    }

    #[test]
    fn test_ensemble_without_collapsed_members_keeps_healthy_members() {
        // A 10-member ensemble with ONE collapsed brick must keep its nine
        // healthy members (without_collapsed_members), instead of dropping the
        // whole verifier to an untrained no-op (which would revert to
        // classifier-only gating).  Fully-collapsed verifiers still fall back
        // to an untrained no-op.
        let mut rng = StdRng::seed_from_u64(1025);
        let positives: Vec<Vec<f32>> = (0..60).map(|_| make_positive_embedding(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..50).map(|_| make_negative_embedding(&mut rng)).collect();
        let pos_seq = make_seq(
            positives,
            crate::audio::embedding_sequence::LabelStratum::Positive,
        );
        let neg_seq = make_seq(
            negatives,
            crate::audio::embedding_sequence::LabelStratum::Negative,
        );
        let mut ensemble = VoiceVerifier::train_ensemble(
            &[pos_seq],
            &[neg_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            CONV_L2_LAMBDA,
        );
        assert!(!ensemble.is_collapsed(), "healthy ensemble not collapsed");

        // Capture each member's standalone score on a fixed probe BEFORE
        // poisoning, so the trimmed mean can be predicted exactly.
        let probe = make_positive_embedding(&mut rng);
        let healthy_member_scores: Vec<f32> = (0..ensemble.ensemble_size())
            .map(|idx| ensemble.member_only(idx).predict(&probe))
            .collect();
        assert_eq!(healthy_member_scores.len(), VERIFIER_ENSEMBLE_SEEDS);

        // Poison the LAST member only.
        *ensemble.ensemble_members.last_mut().expect("members exist") = constant_reject_brick();
        assert!(
            ensemble.is_collapsed(),
            "poisoned ensemble must be flagged collapsed",
        );

        let trimmed = ensemble.without_collapsed_members();
        assert!(trimmed.is_trained(), "healthy members must survive");
        assert_eq!(
            trimmed.ensemble_size(),
            VERIFIER_ENSEMBLE_SEEDS - 1,
            "exactly the collapsed member is dropped",
        );
        // The trimmed mean equals the mean of the surviving healthy members.
        let expected = healthy_member_scores[..VERIFIER_ENSEMBLE_SEEDS - 1]
            .iter()
            .copied()
            .sum::<f32>()
            / (VERIFIER_ENSEMBLE_SEEDS - 1) as f32;
        let trimmed_score = trimmed.predict(&probe);
        assert!(
            (trimmed_score - expected).abs() < 1e-4,
            "trimmed mean must equal the healthy members' mean: \
             trimmed={trimmed_score:.4} expected={expected:.4}",
        );

        // A fully-collapsed verifier (no healthy members) degrades to an
        // untrained no-op.
        let fully_collapsed = constant_reject_brick();
        assert!(fully_collapsed.is_collapsed());
        let resolved = fully_collapsed.without_collapsed_members();
        assert!(
            !resolved.is_trained(),
            "fully collapsed must drop to untrained no-op",
        );
    }

    // ── Additional correctness tests ────────────────────────────────

    #[test]
    fn test_sigmoid_symmetry() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6, "sigmoid(0) != 0.5");
        assert!((sigmoid(10.0) - 1.0).abs() < 1e-4, "sigmoid(10) != ~1.0");
        assert!((sigmoid(-10.0) - 0.0).abs() < 1e-4, "sigmoid(-10) != ~0.0");
    }

    #[test]
    fn test_mean_pool_embeddings_basic() {
        let embs = vec![vec![1.0, 2.0, 3.0], vec![3.0, 4.0, 5.0]];
        let pooled = mean_pool_embeddings(&embs);
        assert_eq!(pooled.len(), 3);
        assert!((pooled[0] - 2.0).abs() < 1e-6);
        assert!((pooled[1] - 3.0).abs() < 1e-6);
        assert!((pooled[2] - 4.0).abs() < 1e-6);

        // Empty input produces empty output.
        assert!(mean_pool_embeddings(&[]).is_empty());

        // Mean-pool a simple 3-frame pattern via mean_pool_window_into.
        let mut window = [0.0f32; VERIFIER_INPUT_DIM];
        for j in 0..VERIFIER_WINDOW_SIZE {
            for i in 0..EMBEDDING_DIM {
                window[j * EMBEDDING_DIM + i] = (j * 10 + i) as f32;
            }
        }
        let mut pooled = [0.0f32; EMBEDDING_DIM];
        mean_pool_window_into(&window, &mut pooled);

        // Frame 0: [0, 1, 2, ..., 95]
        // Frame 1: [10, 11, 12, ..., 105]
        // Frame 2: [20, 21, 22, ..., 115]
        // Mean: [(0+10+20)/3, (1+11+21)/3, ...] = [10, 11, 12, ...]
        for i in 0..EMBEDDING_DIM {
            let correct = ((i + 0) + (i + 10) + (i + 20)) as f32 / 3.0;
            assert!(
                (pooled[i] - correct).abs() < 1e-5,
                "pooled[{i}] = {}, expected {correct}",
                pooled[i],
            );
        }
    }

    #[test]
    fn test_generate_synthetic_negatives_from_positives_basic() {
        let positives: Vec<Vec<f32>> = vec![vec![0.5; 96], vec![0.6; 96], vec![0.4; 96]];
        let negs = generate_synthetic_negatives_from_positives(10, &positives, 1.5, None);
        assert_eq!(negs.len(), 10);
        assert_eq!(negs[0].len(), 96);
        // All values should be finite.
        for emb in &negs {
            for &v in emb {
                assert!(v.is_finite(), "Negative has non-finite value {v}");
            }
        }
        // Negatives should be L2-normalised (unit norm).
        for emb in &negs {
            let norm: f32 = emb.iter().map(|x| x * x).sum();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "Negative embedding is not unit-norm: norm={norm}",
            );
        }

        // Zero count returns empty.
        let zero = generate_synthetic_negatives_from_positives(0, &positives, 1.5, None);
        assert!(zero.is_empty());

        // Empty positives returns empty.
        let empty = generate_synthetic_negatives_from_positives(10, &[], 1.5, None);
        assert!(empty.is_empty());
    }

    #[allow(clippy::cast_precision_loss)]
    #[test]
    fn test_generate_synthetic_negatives_from_positives_per_dim_std() {
        // Two dimensions with very different spreads: dim 0 has tight
        // cluster (low std), dim 1 has wide spread (high std).  The
        // synthetic negatives must reflect this — dim 1 should show
        // proportionally larger perturbations than dim 0.
        let positives: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                let d0 = 0.5; // constant — no variance
                let d1 = 0.5 + (i as f32 - 25.0) / 25.0 * 2.0; // ~N(0.5, 1.0)
                vec![d0, d1]
            })
            .collect();
        let negs = generate_synthetic_negatives_from_positives(200, &positives, 1.0, None);
        assert_eq!(negs.len(), 200);

        // Compute per-dimension std of the generated negatives.
        let mut neg_mean = vec![0.0; 2];
        let mut neg_std = vec![0.0; 2];
        for emb in &negs {
            for (m, &v) in neg_mean.iter_mut().zip(emb.iter()) {
                *m += v;
            }
        }
        for m in &mut neg_mean {
            *m /= negs.len() as f32;
        }
        for emb in &negs {
            for ((s, &v), &m) in neg_std.iter_mut().zip(emb.iter()).zip(neg_mean.iter()) {
                *s += (v - m) * (v - m);
            }
        }
        let n = negs.len() as f32;
        for s in &mut neg_std {
            *s = (*s / n).sqrt();
        }

        // Dim 1 should have significantly larger std than dim 0 (which
        // started from near-constant positives so should remain tight).
        // Note: L2 normalization couples dimensions, so dim 0 picks up
        // some spread from dim 1 — a factor of 2× is still meaningful.
        assert!(
            neg_std[1] > neg_std[0] * 2.0,
            "High-variance dimension should show larger spread: dim0_std={}, dim1_std={}",
            neg_std[0],
            neg_std[1],
        );
    }

    #[test]
    fn test_verifier_empty_training_returns_untrained() {
        // No positive examples → should return untrained.
        let neg_embs = vec![vec![0.0; VERIFIER_INPUT_DIM]];
        let neg_seq = make_seq(
            neg_embs,
            crate::audio::embedding_sequence::LabelStratum::Negative,
        );
        let verifier = VoiceVerifier::train(
            &[],
            &[neg_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            0.001,
            None, // rng_seed (entropy-based)
        );
        assert!(!verifier.is_trained());
    }

    #[test]
    fn test_deterministic_training_same_seed_identical_weights() {
        // ── High-level deterministic check (VoiceVerifier::train) ───
        // Two training runs with the same seed and identical training data
        // must produce identical Conv1D training weights.
        let mut rng = StdRng::seed_from_u64(12345);
        let positives: Vec<Vec<f32>> = (0..40).map(|_| make_positive_embedding(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..40).map(|_| make_negative_embedding(&mut rng)).collect();
        let pos_seq = make_seq(
            positives,
            crate::audio::embedding_sequence::LabelStratum::Positive,
        );
        let neg_seq = make_seq(
            negatives,
            crate::audio::embedding_sequence::LabelStratum::Negative,
        );

        let seed = 42;
        let v1 = VoiceVerifier::train(
            &[pos_seq.clone()],
            &[neg_seq.clone()],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            CONV_L2_LAMBDA,
            Some(seed),
        );
        let v2 = VoiceVerifier::train(
            &[pos_seq],
            &[neg_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            CONV_L2_LAMBDA,
            Some(seed),
        );

        assert!(
            v1.is_trained(),
            "first training produced untrained verifier"
        );
        assert!(
            v2.is_trained(),
            "second training produced untrained verifier"
        );
        // Conv1D deterministic check: weights must be identical with same seed.
        assert_eq!(
            v1.conv_weight, v2.conv_weight,
            "conv_weight differs between deterministic Conv1D training runs"
        );
        assert_eq!(
            v1.conv_bias, v2.conv_bias,
            "conv_bias differs between deterministic Conv1D training runs"
        );
        assert_eq!(
            v1.fc_weight, v2.fc_weight,
            "fc_weight differs between deterministic Conv1D training runs"
        );
        assert_eq!(
            v1.fc_bias, v2.fc_bias,
            "fc_bias differs between deterministic Conv1D training runs"
        );
    }

    #[test]
    fn test_split_negative_sequences_multi_tier_deterministic() {
        // Regression guard (mahbot-1081): the per-tier grouping must be
        // first-appearance-order deterministic.  A HashMap grouping randomized
        // tier-processing order across instances (RandomState per instance),
        // which reordered the sequential rng draw sequence each tier's shuffle
        // consumes — the same seed then produced different splits (and hence
        // different verifier weights) across processes.  The single-tier
        // determinism test above could never catch that; this one exercises
        // multiple source tiers.
        use crate::audio::embedding_sequence::Source;
        let mut infos: Vec<SeqInfo> = Vec::new();
        for (src, n) in [
            (Source::Confusable, 5usize),
            (Source::Unrelated, 7usize),
            (Source::Ambient, 9usize),
            (Source::Synthetic, 11usize),
        ] {
            for i in 0..n {
                infos.push(SeqInfo {
                    start: i,
                    count: 1,
                    is_positive: false,
                    source: src,
                    augmentation_family: None,
                });
            }
        }
        let run = |seed: u64| {
            let mut rng = StdRng::seed_from_u64(seed);
            split_negative_sequences(&infos, &mut rng, true)
        };
        let (tr1, val1) = run(7);
        let (tr2, val2) = run(7);
        assert_eq!(tr1, tr2, "train split differs between same-seed runs");
        assert_eq!(val1, val2, "val split differs between same-seed runs");
        // Sanity: the split partitions all negatives across the tiers.
        assert_eq!(tr1.len() + val1.len(), infos.len());
        assert!(!val1.is_empty(), "multi-tier split produced no validation");
    }

    #[test]
    fn test_verifier_no_cross_utterance_windows() {
        // Two sequences (positive + negative) each shorter than
        // VERIFIER_WINDOW_SIZE (3) → 0 windows from each, but they're in
        // the same training call.  No cross-sequence window should exist
        // (the old combined-flat-list approach would create one window
        // spanning the boundary between them).
        let embs1: Vec<Vec<f32>> = (0..2)
            .map(|i| vec![0.5 + i as f32; EMBEDDING_DIM])
            .collect();
        let embs2: Vec<Vec<f32>> = (0..2)
            .map(|i| vec![-0.5 - i as f32; EMBEDDING_DIM])
            .collect();
        let pos_seq = make_seq(
            embs1,
            crate::audio::embedding_sequence::LabelStratum::Positive,
        );
        let neg_seq = make_seq(
            embs2,
            crate::audio::embedding_sequence::LabelStratum::Negative,
        );

        // With per-sequence windowing, each sequence has 2 frames < 3 → 0 windows each
        // → training gets 0 positive windows + 0 negative windows → untrained.
        let verifier = VoiceVerifier::train(
            &[pos_seq],
            &[neg_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            CONV_L2_LAMBDA,
            Some(42),
        );
        assert!(
            !verifier.is_trained(),
            "Cross-sequence boundary window eliminated — each sequence < WINDOW_SIZE"
        );
    }

    // ── mahbot-1008 tests ────────────────────────────────────────────────

    #[test]
    fn test_verifier_positive_windows_below_minimum_untrained() {
        // Fewer than MIN_POSITIVE_WINDOWS positive windows → untrained no-op
        // (a trained reject-all memorizing a handful of windows is worse than
        // no verifier at all — mahbot-1008 Fix 2).
        let mut rng = StdRng::seed_from_u64(11);
        let positives: Vec<Vec<f32>> = (0..20).map(|_| make_positive_embedding(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..40).map(|_| make_negative_embedding(&mut rng)).collect();
        let pos_seq = make_seq(
            positives,
            crate::audio::embedding_sequence::LabelStratum::Positive,
        );
        let neg_seq = make_seq(
            negatives,
            crate::audio::embedding_sequence::LabelStratum::Negative,
        );

        let verifier = VoiceVerifier::train(
            &[pos_seq],
            &[neg_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            CONV_L2_LAMBDA,
            Some(42),
        );
        assert!(
            !verifier.is_trained(),
            "verifier with {MIN_POSITIVE_WINDOWS}-window minimum must stay untrained \
             when trained on 20 positive windows",
        );
        // Untrained no-op: every frame passes.
        let score = verifier.predict(&[0.5; VERIFIER_INPUT_DIM]);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_verifier_zero_positive_windows_untrained() {
        // The stricter failure mode: positive sequences EXIST but every
        // utterance has < VERIFIER_WINDOW_SIZE frames → zero positive windows.
        // The pre-fix code trained an all-negative reject-all with
        // trained:true; the guard must return untrained instead (mahbot-1008
        // Fix 2, analyst review).
        // Two short positive sequences (2 frames each → 0 windows, since
        // VERIFIER_WINDOW_SIZE = 3) — the zero-positive-windows failure mode.
        let pos_embs: Vec<Vec<f32>> = vec![vec![0.5; EMBEDDING_DIM], vec![1.5; EMBEDDING_DIM]];
        let neg_embs: Vec<Vec<f32>> = (0..40).map(|_| vec![-0.5; EMBEDDING_DIM]).collect();
        let pos_seq = make_seq(
            pos_embs,
            crate::audio::embedding_sequence::LabelStratum::Positive,
        );
        let neg_seq = make_seq(
            neg_embs,
            crate::audio::embedding_sequence::LabelStratum::Negative,
        );

        let verifier = VoiceVerifier::train(
            &[pos_seq],
            &[neg_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            CONV_L2_LAMBDA,
            Some(42),
        );
        assert!(
            !verifier.is_trained(),
            "zero positive windows must yield an untrained no-op, not a trained reject-all",
        );
        // All-negative training must never produce a trained brick wall.
        let score = verifier.predict(&[0.5; VERIFIER_INPUT_DIM]);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_verifier_class_weight_capped() {
        // Extremely imbalanced data must cap the per-positive-window class
        // weight at MAX_CLASS_WEIGHT instead of exploding (~2,208× observed
        // pre-fix) — mahbot-1008 Fix 4.
        let mut rng = StdRng::seed_from_u64(13);
        let positives: Vec<Vec<f32>> = (0..32).map(|_| make_positive_embedding(&mut rng)).collect();
        // 2000 negatives with 1.0 weight → raw ratio ~67× > MAX_CLASS_WEIGHT.
        // (Reduced from 3200 in mahbot-1029 — the cap-triggering ratio keeps
        // >50× with a ~33% margin, cutting ~40% of the training cost.)
        let negatives: Vec<Vec<f32>> = (0..2000)
            .map(|_| make_negative_embedding(&mut rng))
            .collect();
        let pos_seq = make_seq(
            positives,
            crate::audio::embedding_sequence::LabelStratum::Positive,
        );
        let neg_seq = make_seq(
            negatives,
            crate::audio::embedding_sequence::LabelStratum::Negative,
        );

        let (verifier, metrics) = VoiceVerifier::train_with_metrics(
            &[pos_seq],
            &[neg_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            CONV_L2_LAMBDA,
            Some(42),
        );
        assert!(
            verifier.is_trained(),
            "verifier must train at the capped weight"
        );
        let _ = metrics; // training diagnostics are report-only
        // The capped class weight still lets the model discriminate.
        let held_out_pos = make_positive_embedding(&mut rng);
        let held_out_neg = make_negative_embedding(&mut rng);
        assert!(
            verifier.predict(&held_out_pos) > verifier.predict(&held_out_neg),
            "capped-weight verifier must still discriminate",
        );
    }

    #[test]
    fn test_verifier_is_collapsed_detects_constant_reject() {
        // A trained verifier whose feature path is dead (all conv weights zero
        // → output = sigmoid(fc_bias) regardless of input) must be flagged as
        // collapsed (mahbot-1008).  This is the load-time guard that converts
        // already-enrolled users' brick walls into untrained no-ops.
        let collapsed = VoiceVerifier {
            trained: true,
            threshold: DEFAULT_VERIFIER_THRESHOLD,
            activation: VerifierActivation::LeakyReLU,
            conv_weight: vec![0.0; CONV_VERIFIER_OUT * EMBEDDING_DIM * CONV_VERIFIER_KERNEL_SIZE],
            conv_bias: vec![0.0; CONV_VERIFIER_OUT],
            fc_weight: vec![0.0; CONV_VERIFIER_OUT],
            fc_bias: vec![-16.52], // observed pre-fix drift → sigmoid ≈ 6.67e-8
            ensemble_members: Vec::new(),
        };
        assert!(collapsed.is_trained());
        assert!(
            collapsed.is_collapsed(),
            "constant-reject verifier must be flagged as collapsed",
        );

        // A discriminating verifier must NOT be flagged.
        let mut rng = StdRng::seed_from_u64(17);
        let positives: Vec<Vec<f32>> = (0..40).map(|_| make_positive_embedding(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..40).map(|_| make_negative_embedding(&mut rng)).collect();
        let pos_seq = make_seq(
            positives,
            crate::audio::embedding_sequence::LabelStratum::Positive,
        );
        let neg_seq = make_seq(
            negatives,
            crate::audio::embedding_sequence::LabelStratum::Negative,
        );
        let healthy = VoiceVerifier::train(
            &[pos_seq],
            &[neg_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            CONV_L2_LAMBDA,
            Some(42),
        );
        assert!(healthy.is_trained());
        assert!(
            !healthy.is_collapsed(),
            "discriminating verifier must not be flagged as collapsed",
        );

        // An untrained verifier (no-op) is never collapsed.
        assert!(!VoiceVerifier::untrained().is_collapsed());
    }

    #[test]
    fn test_verifier_split_positive_group_holdout() {
        // Positives from two provenance groups: the split must hold out an
        // ENTIRE group for validation (out-of-session validation, mahbot-1008
        // Fix 1) — not an 80/20 per-sequence mix from the same conditions.
        use crate::audio::embedding_sequence::AugmentationFamily;
        use crate::audio::embedding_sequence::Source;
        use crate::audio::embedding_sequence::UtteranceId;

        let mut rng = StdRng::seed_from_u64(19);
        let mut mk_embs = |n: usize| -> Vec<Vec<f32>> {
            (0..n).map(|_| make_positive_embedding(&mut rng)).collect()
        };

        // 6 Enrollment originals × 8 pre-windowed vectors (8 windows each = 48
        // windows) — 288-dim vectors are used directly as pre-windowed data by
        // form_conv1d_sequence_windows.
        // 6 Augmentation(SpeedDown) × 8 pre-windowed vectors (48 windows)
        // 96 positive windows total; holding out either 48-window group leaves
        // 48 ≥ MIN_POSITIVE_WINDOWS in training.
        let mut pos_seqs: Vec<EmbeddingSequence> = Vec::new();
        for i in 0..6 {
            pos_seqs.push(EmbeddingSequence::positive(
                UtteranceId {
                    sequence_index: i,
                    variant_index: 0,
                },
                Source::Enrollment,
                None,
                mk_embs(8),
            ));
            pos_seqs.push(EmbeddingSequence::positive(
                UtteranceId {
                    sequence_index: i,
                    variant_index: 1,
                },
                Source::Augmentation,
                Some(AugmentationFamily::SpeedDown),
                mk_embs(8),
            ));
        }
        // 6 negative sequences × 8 frames.
        let neg_seqs: Vec<EmbeddingSequence> = (0..6)
            .map(|_| {
                EmbeddingSequence::negative(
                    UtteranceId {
                        sequence_index: 0,
                        variant_index: 0,
                    },
                    Source::Synthetic,
                    None,
                    (0..8).map(|_| make_negative_embedding(&mut rng)).collect(),
                )
            })
            .collect();

        // The preferred holdout group is (Enrollment, None) — the originals.
        let prepared =
            prepare_training_data(&pos_seqs, &neg_seqs, None, form_conv1d_sequence_windows)
                .expect("training data must prepare");
        let (tr_w, tr_l, _tr_wt, val_w, val_l, _val_wt, kind) = split_train_val(
            &prepared.3,
            &prepared.0,
            &prepared.1,
            &prepared.2,
            &mut StdRng::seed_from_u64(19),
            true,
        );

        assert_eq!(
            kind,
            SplitKind::GroupHoldout,
            "split must use group holdout"
        );
        let n_tr_pos = tr_l.iter().filter(|&&l| l > 0.5).count();
        let n_val_pos = val_l.iter().filter(|&&l| l > 0.5).count();
        // The entire 48-window Enrollment group is held out; training keeps the
        // 48 SpeedDown windows.
        assert_eq!(
            n_val_pos, 48,
            "held-out group must be the full 48 originals"
        );
        assert_eq!(n_tr_pos, 48, "training keeps the 48 augmented windows");
        assert_eq!(tr_w.len() + val_w.len(), 96 + 48, "no windows lost");
    }

    #[test]
    fn test_verifier_split_no_leaky_per_window_fallback() {
        // All positives from one provenance group with too few sequences for a
        // per-sequence split: the split must yield NO validation data rather
        // than the pre-fix leaky per-window fallback (which validated on
        // windows from the same sequences as training — mahbot-1008 Fix 1).
        let mut rng = StdRng::seed_from_u64(23);
        let positives: Vec<Vec<f32>> = (0..40).map(|_| make_positive_embedding(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..40).map(|_| make_negative_embedding(&mut rng)).collect();
        let pos_seq = make_seq(
            positives,
            crate::audio::embedding_sequence::LabelStratum::Positive,
        );
        let neg_seq = make_seq(
            negatives,
            crate::audio::embedding_sequence::LabelStratum::Negative,
        );

        let prepared =
            prepare_training_data(&[pos_seq], &[neg_seq], None, form_conv1d_sequence_windows)
                .expect("training data must prepare");
        let (tr_w, _tr_l, _tr_wt, val_w, _val_l, _val_wt, kind) = split_train_val(
            &prepared.3,
            &prepared.0,
            &prepared.1,
            &prepared.2,
            &mut StdRng::seed_from_u64(23),
            true,
        );
        assert_eq!(
            kind,
            SplitKind::None,
            "no validation data, no leaky fallback"
        );
        assert!(
            val_w.is_empty(),
            "validation must be empty (no per-window leak)"
        );
        assert_eq!(tr_w.len(), 80, "all windows stay in training");
    }

    #[test]
    fn test_verifier_split_per_sequence_below_minimum_uses_all_data() {
        // When the per-sequence split would leave fewer than
        // MIN_POSITIVE_WINDOWS positive windows in training, ALL positives
        // must stay in training with empty validation — the
        // prepare_training_data guard checks the pre-split TOTAL, but the
        // split can carve training below the minimum (mahbot-1011).
        let mut rng = StdRng::seed_from_u64(31);
        // 5 positive sequences × 6 windows = 30 total positive windows —
        // passes the pre-split MIN_POSITIVE_WINDOWS guard, but all from one
        // provenance group (Enrollment, None), so no group holdout is
        // possible.  A 20% per-sequence split would send 1 sequence (6
        // windows) to validation, leaving 24 < MIN_POSITIVE_WINDOWS in
        // training.
        let pos_seqs: Vec<EmbeddingSequence> = (0..5)
            .map(|_| {
                make_seq(
                    (0..6).map(|_| make_positive_embedding(&mut rng)).collect(),
                    crate::audio::embedding_sequence::LabelStratum::Positive,
                )
            })
            .collect();
        // 4 negative sequences × 8 windows = 32 negative windows.
        let neg_seqs: Vec<EmbeddingSequence> = (0..4)
            .map(|_| {
                make_seq(
                    (0..8).map(|_| make_negative_embedding(&mut rng)).collect(),
                    crate::audio::embedding_sequence::LabelStratum::Negative,
                )
            })
            .collect();

        let prepared =
            prepare_training_data(&pos_seqs, &neg_seqs, None, form_conv1d_sequence_windows)
                .expect("training data must prepare");
        let (tr_w, tr_l, _tr_wt, val_w, _val_l, _val_wt, kind) = split_train_val(
            &prepared.3,
            &prepared.0,
            &prepared.1,
            &prepared.2,
            &mut StdRng::seed_from_u64(31),
            true,
        );
        assert_eq!(
            kind,
            SplitKind::None,
            "split below the positive-window minimum must fall back to all-data \
             training (no validation, no leaky fallback)"
        );
        assert!(val_w.is_empty(), "validation must be empty");
        let n_tr_pos = tr_l.iter().filter(|&&l| l > 0.5).count();
        assert_eq!(n_tr_pos, 30, "all 30 positive windows stay in training");
        assert_eq!(tr_w.len(), 62, "all 30 positive + 32 negative windows kept");
    }

    #[test]
    fn test_verifier_legacy_json_defaults_to_leaky_relu() {
        // A verifier JSON persisted before mahbot-1008 has no `activation`
        // field; it must deserialize with the LeakyReLU default and remain
        // usable (mahbot-1008 persistence compatibility).
        let json = r#"{
            "conv_weight": [],
            "conv_bias": [],
            "fc_weight": [],
            "fc_bias": [],
            "threshold": 0.948,
            "trained": false
        }"#;
        let verifier: VoiceVerifier = serde_json::from_str(json).expect("legacy JSON loads");
        assert_eq!(verifier.activation, VerifierActivation::LeakyReLU);
        assert!(!verifier.is_trained());
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_verifier_train_sequences() {
        // Eight positive sequences + eight negative sequences each with enough
        // frames to form windows → trained verifier accepts positives and
        // rejects negatives.  Counts are ≥ MIN_POSITIVE_WINDOWS (30) after the
        // per-sequence 80/20 split (mahbot-1008 Fix 2 guard).
        let mut rng = StdRng::seed_from_u64(42);

        // Positive sequences: 8 frames → 6 stride-1 windows each → 48 total
        let pos_seqs: Vec<EmbeddingSequence> = (0..8)
            .map(|_| {
                let embs: Vec<Vec<f32>> =
                    (0..8).map(|_| make_positive_embedding(&mut rng)).collect();
                make_seq(
                    embs,
                    crate::audio::embedding_sequence::LabelStratum::Positive,
                )
            })
            .collect();
        // Negative sequences: 8 frames → 6 stride-1 windows each → 48 total
        let neg_seqs: Vec<EmbeddingSequence> = (0..8)
            .map(|_| {
                let embs: Vec<Vec<f32>> =
                    (0..8).map(|_| make_negative_embedding(&mut rng)).collect();
                make_seq(
                    embs,
                    crate::audio::embedding_sequence::LabelStratum::Negative,
                )
            })
            .collect();

        let verifier = VoiceVerifier::train(
            &pos_seqs,
            &neg_seqs,
            None, // no per-negative weights
            DEFAULT_VERIFIER_THRESHOLD,
            CONV_L2_LAMBDA,
            Some(42),
        );

        assert!(
            verifier.is_trained(),
            "Multi-sequence verifier must be trained"
        );

        // Verify held-out positive and negative.
        let held_out_pos = make_positive_embedding(&mut rng);
        let score_pos = verifier.predict(&held_out_pos);
        assert!(
            score_pos >= 0.5,
            "Verifier should accept positive embedding (score >= 0.5), got score={score_pos:.4}",
        );

        let held_out_neg = make_negative_embedding(&mut rng);
        let score_neg = verifier.predict(&held_out_neg);
        assert!(
            score_neg < 0.5,
            "Verifier should reject negative embedding (score < 0.5), got score={score_neg:.4}",
        );

        // ── Cache-weighted sequences ────────────────────────────────
        // Simulates production cache path: 4-tier per-sequence weights
        // (confusable=15×, unrelated=10×, ambient=1×) to validate the
        // dynamic class_weight formula (mahbot-993).  The non-uniform weights
        // mean the class_weight should be meaningfully higher than the raw
        // n_neg/n_pos ratio; the verifier must still discriminate.
        let mut rng2 = StdRng::seed_from_u64(99);
        let cache_pos: Vec<Vec<f32>> = (0..40)
            .map(|_| make_positive_embedding(&mut rng2))
            .collect();
        let cache_pos_seq = make_seq(
            cache_pos,
            crate::audio::embedding_sequence::LabelStratum::Positive,
        );

        let neg_confusable: Vec<Vec<f32>> = (0..10)
            .map(|_| make_negative_embedding(&mut rng2))
            .collect();
        let neg_unrelated: Vec<Vec<f32>> = (0..10)
            .map(|_| make_negative_embedding(&mut rng2))
            .collect();
        let neg_ambient: Vec<Vec<f32>> = (0..10)
            .map(|_| make_negative_embedding(&mut rng2))
            .collect();

        let cache_negatives = [
            make_seq(
                neg_confusable,
                crate::audio::embedding_sequence::LabelStratum::Negative,
            ),
            make_seq(
                neg_unrelated,
                crate::audio::embedding_sequence::LabelStratum::Negative,
            ),
            make_seq(
                neg_ambient,
                crate::audio::embedding_sequence::LabelStratum::Negative,
            ),
        ];

        let per_neg_weights = vec![
            CONFUSABLE_UPWEIGHT, // 15.0 — confusable tier
            UNRELATED_UPWEIGHT,  // 10.0 — unrelated tier
            1.0,                 //       — ambient tier
        ];

        let cache_verifier = VoiceVerifier::train(
            &[cache_pos_seq],
            &cache_negatives,
            Some(&per_neg_weights),
            DEFAULT_VERIFIER_THRESHOLD,
            CONV_L2_LAMBDA,
            Some(42),
        );
        assert!(
            cache_verifier.is_trained(),
            "Cache-weighted verifier must be trained",
        );

        // Verify discrimination with per-negative-weights (mahbot-993).
        // For this distribution (40 pos windows, 30 neg windows with weights
        // [15.0, 10.0, 1.0]), the dynamic formula gives
        // class_weight ≈ (10×15 + 10×10 + 10×1)/40 = 6.5 (vs raw ratio 0.75).
        // The model must still produce positive scores above negative scores.
        let held_out_pos = make_positive_embedding(&mut rng2);
        let held_out_neg = make_negative_embedding(&mut rng2);
        let score_pos = cache_verifier.predict(&held_out_pos);
        let score_neg = cache_verifier.predict(&held_out_neg);
        assert!(
            score_pos > score_neg,
            "Weighted verifier must discriminate: pos={score_pos:.4} neg={score_neg:.4}",
        );
        assert!(
            score_pos >= 0.5,
            "Weighted verifier should accept positive (score >= 0.5), got {score_pos:.4}",
        );
    }

    // ── calibrate_verifier_threshold tests (mahbot-997) ─────────────────

    #[test]
    fn test_calibrate_threshold_basic_youden_selection() {
        // Positives cluster at 0.9, negatives cluster at 0.3.
        // With λ=2.0, the optimal threshold should be between 0.3 and 0.9.
        let pos = vec![0.91, 0.92, 0.89, 0.93, 0.90, 0.88, 0.91, 0.90];
        let neg = vec![0.31, 0.29, 0.32, 0.28, 0.30, 0.27, 0.33, 0.29];
        let result = calibrate_verifier_threshold(&pos, &neg, 0.5);

        // Optimal threshold should be >= 0.33 (all neg < 0.33) and <= 0.88
        // (all pos >= 0.88), i.e., somewhere in (0.33, 0.88).
        assert!(
            result > 0.33 && result <= 0.88,
            "Optimal threshold should separate pos ({:.2}..{:.2}) from neg ({:.2}..{:.2}), got {result:.4}",
            pos.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
            pos.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
            neg.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
            neg.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
        );
    }

    #[test]
    fn test_calibrate_threshold_tpr_constraint_enforced() {
        // All negatives at 0.1, positives at three levels:
        // 0.95 (4 samples), 0.85 (4 samples), 0.50 (4 samples).
        let pos = vec![
            0.95, 0.96, 0.94, 0.95, // high cluster
            0.85, 0.86, 0.84, 0.85, // mid cluster
            0.50, 0.51, 0.49, 0.50, // low cluster
        ];
        let neg = vec![0.10, 0.11, 0.09, 0.10, 0.12, 0.08, 0.10, 0.11];
        let result = calibrate_verifier_threshold(&pos, &neg, 0.5);

        // At threshold 0.52: TPR = 12/12 = 1.0, FPR = 0/8 = 0.0, Youden = 1.0.
        // At threshold 0.87: TPR = 4/12 = 0.33 < 0.90 → rejected.
        // With `>=`, the algorithm prefers the highest threshold with maximal
        // Youden, which is 0.98 (last threshold with TPR=1.0, FPR=0.0 before
        // the first positive at 0.49 drops out at 0.50).
        // At threshold 0.50: TPR = 8/12 = 0.667 < 0.90 → rejected.
        // The optimal is 0.49 (last threshold at which all 12 positives ≥ T).
        // Since we iterate in 0.01 steps, this is 0.49.
        assert!(
            (result - 0.49).abs() < 0.02,
            "Expected threshold ~0.49 (highest with TPR=1.0, FPR=0.0), got {result:.4}",
        );
    }

    #[test]
    fn test_calibrate_threshold_sparse_positives_falls_back() {
        // Fewer than CALIBRATION_MIN_SAMPLES (5) positives.
        let pos = vec![0.9, 0.8, 0.7, 0.6];
        let neg = vec![0.3, 0.2, 0.1, 0.3, 0.2, 0.1];
        let result = calibrate_verifier_threshold(&pos, &neg, 0.75);
        assert!(
            (result - 0.75).abs() < 1e-6,
            "Sparse positives should fall back to default 0.75, got {result:.4}",
        );
    }

    #[test]
    fn test_calibrate_threshold_sparse_negatives_falls_back() {
        // Fewer than CALIBRATION_MIN_SAMPLES (5) negatives.
        let pos = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4];
        let neg = vec![0.3, 0.2, 0.1, 0.3];
        let result = calibrate_verifier_threshold(&pos, &neg, 0.60);
        assert!(
            (result - 0.60).abs() < 1e-6,
            "Sparse negatives should fall back to default 0.60, got {result:.4}",
        );
    }

    #[test]
    fn test_calibrate_threshold_lambda_penalizes_false_accepts() {
        // Score distribution where a mid-range threshold gives the same
        // TPR as a low threshold but lower FPR.
        let pos = vec![
            0.95, 0.94, 0.93, 0.92, 0.91, // 5 high pos
            0.70, 0.69, 0.68, 0.67, 0.66, // 5 low pos
        ];
        let neg = vec![
            0.75, 0.74, 0.73, 0.72, 0.71, // 5 high neg
            0.10, 0.09, 0.08, 0.07, 0.06, // 5 low neg
        ];

        // At threshold 0.71: TPR = 5/10 = 0.5 < 0.90 → rejected.
        // At threshold 0.66: TPR = 10/10 = 1.0, FPR = 5/10 = 0.5, Youden = 1.0 - 2*0.5 = 0.0.
        // At threshold 0.76: TPR = 0/10 = 0.0 < 0.90 → rejected.
        // Among valid thresholds (0.0..0.66), Youden increases as FPR drops.
        // At threshold 0.66: Youden = 0.0.
        // At threshold 0.11: FPR = 5/10 = 0.5, Youden = 0.0.
        // With `>=`, the highest threshold with Youden=0.0 is 0.66 (the last
        // at which TPR=1.0 before low-pos at 0.66 drops out).
        let result = calibrate_verifier_threshold(&pos, &neg, 0.5);
        assert!(
            (result - 0.66).abs() < 0.02,
            "Expected threshold ~0.66 (highest with TPR=1.0), got {result:.4}",
        );
    }

    #[test]
    fn test_calibrate_threshold_all_same_scores_prefers_highest() {
        // All scores identical (0.5). At any threshold ≤ 0.5, TPR=1.0, FPR=1.0,
        // Youden = -1.0. At threshold > 0.5, TPR=0.0 < 0.90 → rejected.
        // With `>=`, the algorithm prefers the highest threshold with equal
        // Youden, which is 0.50.
        let pos = vec![0.5; 8];
        let neg = vec![0.5; 8];
        let result = calibrate_verifier_threshold(&pos, &neg, 0.6);

        assert!(
            (result - 0.50).abs() < 1e-6,
            "All identical scores should give threshold 0.50 (last with TPR=1.0), got {result:.4}",
        );
    }

    #[test]
    fn test_calibrate_threshold_perfect_separation() {
        // Perfectly separated: all positives at 0.99, all negatives at 0.01.
        let pos = vec![0.99; 10];
        let neg = vec![0.01; 10];
        let result = calibrate_verifier_threshold(&pos, &neg, 0.5);

        // Any threshold in (0.01, 0.99) gives TPR = 1.0, FPR = 0.0, Youden = 1.0.
        // With `>=`, the algorithm picks the last threshold with maximal Youden.
        // At threshold 0.98: TPR = 1.0, FPR = 0.0, Youden = 1.0.
        // At threshold 0.99: TPR = 1.0 (all >= 0.99), FPR = 0.0, Youden = 1.0.
        // At threshold 1.00: TPR = 0/10 = 0.0 < 0.90 → rejected.
        // So the highest valid threshold is 0.99.
        assert!(
            (result - 0.99).abs() < 1e-6,
            "Perfect separation should give threshold 0.99 (highest with TPR=1.0, FPR=0.0), got {result:.4}",
        );
    }

    #[test]
    fn test_calibrate_threshold_overlapping_distributions() {
        // Positives and negatives overlap fully — all thresholds have TPR = FPR,
        // so Youden = TPR - 2×FPR = -TPR is constant for all valid thresholds.
        // This exercises the tiebreaker (preferring higher thresholds for equal
        // Youden) rather than the Youden optimization itself.
        let pos = vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
        let neg = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2];

        // At threshold 0.10: TPR = 8/8 = 1.0, FPR = 8/8 = 1.0, Youden = -1.0
        // At threshold 0.20: TPR = 7/8 = 0.875 < 0.90 → rejected? Wait...
        //   pos >= 0.20: [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2] = 7/8 = 0.875 < 0.90 → rejected.
        // At threshold 0.10: TPR = 8/8 = 1.0, FPR = 8/8 = 1.0, Youden = -1.0
        // At threshold 0.11: same, all >= 0.11.
        // At threshold 0.20: The pos element at 0.1 is < 0.20, so TPR = 0.875 < 0.9 → rejected.
        // So only thresholds ≤ 0.10 are valid.
        // At threshold 0.0: TPR = 1.0, FPR = 1.0, Youden = -1.0.
        // At threshold 0.10: TPR = 1.0, FPR = 1.0, Youden = -1.0.
        // The algorithm picks the highest valid threshold = 0.10.
        let result = calibrate_verifier_threshold(&pos, &neg, 0.7);

        assert!(
            (result - 0.10).abs() < 0.02,
            "Overlapping distributions: expected threshold ~0.10, got {result:.4}",
        );
    }
}
