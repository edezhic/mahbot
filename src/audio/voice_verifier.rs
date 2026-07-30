//! Verifier for wake word false-trigger suppression.
//!
//! Implements a lightweight second-stage classifier that runs AFTER the
//! Conv1D classifier fires, as an additional AND gate.
//!
//! ## Primary architecture (mahbot-995): Conv1D
//!
//! Conv1D(96→2, k=3, padding=1) → ReLU → AdaptiveAvgPool → Linear(2→1) → Sigmoid.
//! ~581 trainable parameters. Preserves temporal structure across the 3-frame
//! window (288-dim concatenated input, no mean-pooling).
//!
//! ## Legacy architecture (mahbot-901): Logistic regression
//!
//! 97-parameter logistic regression on temporally mean-pooled 96-dim embeddings
//! (L2 regularization removed in mahbot-994). Mean-pools the 3-frame window to
//! 96-dim before L2-norm, scaler, and linear+sigmoid.
//!
//! ## Backward compatibility
//!
//! Old logistic-format models (serialized before mahbot-995) deserialize with
//! `arch: Logistic` and continue working through the logistic path. New training
//! produces `arch: Conv1D` with Conv1D weight fields. `predict()` dispatches
//! based on `arch` via `predict_conv1d()` or `predict_logistic()`.
//!
//! When not trained, the verifier acts as a no-op (all frames pass).
//!
//! # Architecture
//!
//! Training pipeline: per-frame embeddings → windowing (concatenated 288-dim
//! for Conv1D, mean-pooled 96-dim for logistic) → L2-norm → train (Adam for
//! Conv1D, SGD for logistic).  Inference is ~3μs per frame.  The StandardScaler
//! formerly used in the logistic path was removed in mahbot-996 (it is
//! mathematically redundant on L2-normalized embeddings and caused OOD
//! score-underflow vulnerability).
//!
//! ## Training data
//!
//! - **Positive examples**: 3-frame stride-1 windows formed from enrollment
//!   utterance per-frame embeddings.
//! - **Negative examples**: Synthetic Gaussian noise (bootstrapping) or
//!   hard-negative embeddings collected from near-miss frames during detection.
//! - **Confusable negatives**: Pre-computed near-miss phrase embeddings (e.g.
//!   "hey map bot", "day mahbot") with 15× higher per-example weight during
//!   training so the verifier learns to reject confusable phrases.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::audio::embedding_sequence::EmbeddingSequence;
use crate::{EMBEDDING_DIM, VERIFIER_INPUT_DIM, VERIFIER_WINDOW_SIZE};

/// Default decision threshold for the verifier.
///
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
/// ## Two-tier ceiling escalation plan
///
/// If E2E benchmarks show the 4.503 `ADAPTIVE_CEILING` is too aggressive
/// (excessive false rejects), escalate to 5.5.  The escalation trigger is
/// when the per-utterance adaptive threshold trajectory (tracked via
/// `DetectionInstrumentation.adaptive_threshold_trajectory`) shows the
/// ceiling is the active limiting factor on detection rate.
///
/// **Previously:** 0.60 (streaming, mahbot-890), 0.50 (mahbot-882),
/// 0.4 (mahbot-853), 0.6 (mahbot-829), 0.5 (mahbot-797), 0.3 (mahbot-788).
///
/// ⚠ **If changing this constant**, re-calibrate the 1.58× multiplier by
pub(crate) const DEFAULT_VERIFIER_THRESHOLD: f32 = 0.948;

/// L2 regularization strength (lambda).
///
/// Set to 0.0 (disabled) as of mahbot-994.
///
/// ## Rationale
///
/// The convex logistic regression with 97 parameters is trained on thousands of
/// examples with multiple orthogonal forms of regularization:
///
/// - **Early stopping** (patience=100 on validation loss) stops training before
///   overfitting can occur.
/// - **Class weighting** (mahbot-993) ensures balanced gradient signal without
///   needing additional penalties.
/// - **Tier-based example weighting** (Ambient=1×, Unrelated=10×,
///   Confusable=15×) provides per-example importance without global shrinkage.
/// - **L2-normalized input features** bound the feature magnitudes on the
///
/// Previously set to 0.001 (mahbot-949) after the original 0.01 was found
/// to collapse weights to zero.  However, at convergence the BCE gradient
/// approaches zero while the L2 gradient (λ·w) is always present, continuing to
/// shrink weights post-convergence even with λ=0.001.  With plain SGD (no
/// adaptive optimizer), L2 is significantly stronger than the same λ with
/// Adam (used by the Conv1D classifier at λ=0.0001).
///
/// Empirical testing confirmed that removing L2 entirely (λ=0) allows the
/// corrected gradient signals from the dynamic class_weight formula to operate
/// without competing weight decay, producing non-zero weights and a bias that
/// is not forced to extreme negative values.
pub(crate) const L2_LAMBDA: f32 = 0.0;

/// Learning rate for logistic regression SGD training (backward-compat only, mahbot-995).
#[cfg(test)]
pub(crate) const LOGISTIC_LEARNING_RATE: f32 = 0.01;

/// Maximum iterations for logistic regression training (backward-compat only, mahbot-995).
#[cfg(test)]
pub(crate) const LOGISTIC_MAX_ITER: usize = 1000;

/// Fraction of sequences held out for validation (80/20 train/val split).
///
/// Split is per-sequence (avoiding data leakage from overlapping windows) with
/// stratification preserving both pos:neg ratio AND negative tier proportions
/// (confusable/unrelated/ambient/owner).  Falls back to per-window random split
/// when per-sequence split produces insufficient validation data.
pub(crate) const VALIDATION_SPLIT: f32 = 0.2;

/// Early stopping patience for logistic verifier (backward-compat only, mahbot-995).
/// The Conv1D path uses its own [`CONV_EARLY_STOP_PATIENCE`].
#[cfg(test)]
pub(crate) const EARLY_STOP_PATIENCE: usize = 100;

/// Log training and validation loss every N iterations.
const LOG_LOSS_INTERVAL: usize = 50;

// ── Conv1D verifier training hyperparameters (mahbot-995) ────────────────
//
// Matches the Conv1D classifier's proven values (wake_word_classifier.rs)
// for the same small-dataset (<200 positive windows) regime.

/// Learning rate for Conv1D verifier Adam training.
const CONV_LEARNING_RATE: f32 = 0.001;
/// L2 regularization strength for Conv1D verifier.
const CONV_L2_LAMBDA: f32 = 0.0001;
/// Batch size for Conv1D verifier mini-batch training.
const CONV_BATCH_SIZE: usize = 32;
/// Maximum training epochs for Conv1D verifier.
const CONV_MAX_EPOCHS: usize = 100;
/// Early stopping patience for Conv1D verifier.
const CONV_EARLY_STOP_PATIENCE: usize = 15;
/// Conv1D output channels for verifier (96→CONV_VERIFIER_OUT).
const CONV_VERIFIER_OUT: usize = 2;
/// Conv1D kernel size for verifier.
const CONV_VERIFIER_KERNEL_SIZE: usize = 3;

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
/// `cargo bench --bench voice_pipeline_e2e`.  Adjust in source and
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

/// Minimum standard deviation when computing StandardScaler (mahbot-996).
///
/// Prevents division by (near) zero for dimensions with near-zero variance in
/// the training data.  When applied to out-of-distribution voice families,
/// unscaled near-zero std dimensions can produce extreme z-scores that cause
/// sigmoid underflow to exactly 0.0.
///
/// The value 1e-3 is more conservative than the 1e-6 used in
/// [`generate_synthetic_negatives_from_positives`] because scaler division
/// (z-scoring) is more sensitive to near-zero values than noise addition.
/// With L2-normalized embeddings on the unit sphere (dim values in [-1, 1]),
/// a std floor of 1e-3 bounds the z-score to at most ±2000 per dimension,
/// which prevents the worst-case logit explosions when weights are bounded.
///
/// ## Precedent
///
/// The same `.max(N)` pattern is used in [`generate_synthetic_negatives_from_positives`]
/// (`.max(1e-6)`) and throughout the wake word pipeline for numerical stability.
#[cfg(test)]
const SCALER_STD_MIN: f32 = 1e-3;

/// Architecture variant for the verifier (mahbot-995).
///
/// - [`Logistic`](VerifierArch::Logistic): 97-parameter logistic regression on
///   mean-pooled 96-dim embeddings (legacy, mahbot-901). Backward compat only
///   — new training produces [`Conv1D`](VerifierArch::Conv1D).
/// - [`Conv1D`](VerifierArch::Conv1D): Small Conv1D(96→2, k=3) with 581
///   parameters operating on 288-dim concatenated windows (mahbot-995).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum VerifierArch {
    /// 97-parameter logistic regression (legacy, mahbot-901).
    #[default]
    Logistic,
    /// Conv1D(96→2, k=3) with 581 parameters (mahbot-995).
    Conv1D,
}

/// Verifier for wake word false-trigger suppression (second-stage AND gate).
///
/// ## Conv1D architecture (mahbot-995, default for new training)
///
/// Conv1D(96→2, k=3, padding=1) → ReLU → AdaptiveAvgPool → Linear(2→1) → Sigmoid.
/// ~581 trainable parameters operating on 288-dim concatenated windows (no
/// mean-pooling). Preserves temporal structure for better confusable rejection.
///
/// ## Logistic architecture (legacy, mahbot-901)
///
/// 97-parameter logistic regression on temporally mean-pooled 96-dim embeddings.
/// Mean-pools the 3-frame window to 96-dim before L2-norm and linear+sigmoid.
///
/// As of mahbot-996, the StandardScaler is no longer fitted during training.
/// It is mathematically redundant on L2-normalized embeddings — the affine
/// transform can be absorbed into the weights and bias. Removing it eliminates
/// the OOD score-underflow vulnerability where near-zero std dimensions
/// produce extreme z-scores.
///
/// Old persisted models with scaler data continue to be applied during
/// inference for backward compatibility. Re-enrollment after this fix
/// produces scaler-free models that are immune to the underflow issue.
///
/// ## Backward compatibility
///
/// Old logistic-format models deserialize with `arch: Logistic` (via
/// `#[serde(default)]` on the `arch` field) and continue working through the
/// logistic path. New training produces `arch: Conv1D`.
///
/// When `trained` is `false`, the verifier is a no-op (all frames pass with
/// score 1.0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceVerifier {
    /// Architecture variant (Logistic or Conv1D).
    #[serde(default)]
    pub arch: VerifierArch,

    /// Logistic regression weights (96-dim, mahbot-901).
    #[serde(default)]
    pub weights: Vec<f32>,
    /// Logistic regression bias.
    #[serde(default)]
    pub bias: f32,

    /// StandardScaler mean (96-dim). Empty when scaling is not used.
    #[serde(default)]
    pub scaler_mean: Vec<f32>,
    /// StandardScaler std (96-dim). Empty when scaling is not used.
    #[serde(default)]
    pub scaler_std: Vec<f32>,

    /// Conv1D weight: [CONV_VERIFIER_OUT, EMBEDDING_DIM, kernel_size] = [2, 96, 3] = 576.
    #[serde(default)]
    pub conv_weight: Vec<f32>,
    /// Conv1D bias: [CONV_VERIFIER_OUT] = [2].
    #[serde(default)]
    pub conv_bias: Vec<f32>,
    /// FC weight: [CONV_VERIFIER_OUT] → [1] = 2 elements.
    #[serde(default)]
    pub fc_weight: Vec<f32>,
    /// FC bias: [1] = 1 element.
    #[serde(default)]
    pub fc_bias: Vec<f32>,

    /// Decision threshold. Frames with a score below this are suppressed.
    #[serde(default = "default_verifier_threshold")]
    pub threshold: f32,
    /// Whether this verifier has been trained with positive + negative data.
    #[serde(default)]
    pub trained: bool,
}

fn default_verifier_threshold() -> f32 {
    DEFAULT_VERIFIER_THRESHOLD
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
}

impl VoiceVerifier {
    /// Create an untrained verifier (no-op: all frames pass).
    ///
    /// Uses [`VerifierArch::Conv1D`] to match the architecture produced by
    /// [`train`](Self::train) (mahbot-995).  An untrained verifier always returns
    /// `1.0` from [`predict`](Self::predict) regardless of arch, so this is
    /// purely for consistency (the arch field of an untrained verifier is never
    /// used for inference).
    #[must_use]
    pub fn untrained() -> Self {
        Self {
            arch: VerifierArch::Conv1D,
            weights: Vec::new(),
            bias: 0.0,
            scaler_mean: Vec::new(),
            scaler_std: Vec::new(),
            conv_weight: Vec::new(),
            conv_bias: Vec::new(),
            fc_weight: Vec::new(),
            fc_bias: Vec::new(),
            threshold: DEFAULT_VERIFIER_THRESHOLD,
            trained: false,
        }
    }

    /// Returns `true` if this verifier has been trained and is ready for
    /// inference.
    ///
    /// Validates architecture-specific weights:
    /// - Logistic: 96-dim weights, optional scaler, finite bias
    /// - Conv1D: 576-dim conv_weight + 2-dim conv_bias + 2-dim fc_weight + 1-dim fc_bias
    #[must_use]
    pub fn is_trained(&self) -> bool {
        if !self.trained {
            return false;
        }
        match self.arch {
            VerifierArch::Logistic => {
                // Must have 96-dim weights.
                if self.weights.len() != EMBEDDING_DIM {
                    return false;
                }
                // If scaler is present, it must be at 96-dim and both mean and std
                // must be populated.  Empty scaler is OK (inference skips scaling).
                let has_mean = !self.scaler_mean.is_empty();
                let has_std = !self.scaler_std.is_empty();
                if has_mean != has_std {
                    return false; // partial scaler
                }
                if has_mean
                    && (self.scaler_mean.len() != EMBEDDING_DIM
                        || self.scaler_std.len() != EMBEDDING_DIM)
                {
                    return false;
                }
                // Bias must be finite.
                self.bias.is_finite()
            }
            VerifierArch::Conv1D => {
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
        }
    }

    /// Predict the probability that the given window is a genuine wake word.
    ///
    /// For [`VerifierArch::Conv1D`]: requires 288-dim input (3 concatenated
    /// 96-dim embeddings). Panics if input is not 288-dim.
    ///
    /// For [`VerifierArch::Logistic`]: accepts either 288-dim (mean-pools
    /// internally to 96-dim) or 96-dim (already pooled, e.g. from training
    /// diagnostics) input.
    ///
    /// Returns a score in `[0.0, 1.0]`. When untrained, always returns `1.0`
    /// (no-op — all frames pass).
    #[must_use]
    pub fn predict(&self, embedding: &[f32]) -> f32 {
        if !self.is_trained() {
            return 1.0;
        }

        match self.arch {
            VerifierArch::Conv1D => predict_conv1d(
                embedding,
                &self.conv_weight,
                &self.conv_bias,
                &self.fc_weight,
                &self.fc_bias,
            ),
            VerifierArch::Logistic => {
                // Logistic regression on mean-pooled 96-dim embeddings (mahbot-901).
                // Accepts either 288-dim (mean-pools internally) or 96-dim
                // (already pooled, e.g. from training diagnostics).
                predict_logistic(
                    embedding,
                    &self.weights,
                    self.bias,
                    &self.scaler_mean,
                    &self.scaler_std,
                )
            }
        }
    }

    /// Train a new verifier from positive and negative
    /// [`EmbeddingSequence`](crate::audio::embedding_sequence::EmbeddingSequence)
    /// inputs.  Trains a Conv1D classifier with L2 regularization (mahbot-995).
    ///
    /// Windows are formed **within** each sequence independently (never across
    /// sequences), preventing the cross-utterance window contamination that
    /// existed when training operated on flat `&[Vec<f32>]` lists (mahbot-902).
    /// Each window is 3 embeddings (288-dim, not mean-pooled) to preserve
    /// temporal structure.  Windows are L2-normalized before training
    /// (mahbot-870).
    ///
    /// # Arguments
    ///
    /// * `positive_sequences` — [`EmbeddingSequence`] values from enrollment
    ///   utterances (label = `Positive`).  Each sequence's embeddings form
    ///   windows independently; no windows cross between sequences.
    /// * `negative_sequences` — [`EmbeddingSequence`] values from non-wake-word
    ///   audio (label = `Negative`), e.g., confusable phrases, unrelated speech,
    ///   ambient noise, or synthetic negatives.
    /// * `per_negative_sequence_weights` — Optional per-sequence weights
    ///   for negative sequences only (used to upweight confusable near-miss
    ///   phrases).  When `Some(weights)`, `weights.len()` must equal
    ///   `negative_sequences.len()`.  Positives are weighted by an automatic
    ///   class weight that balances the **total effective gradient** from both
    ///   classes: `Σ(per_neg_weight_i × window_count_i) / n_pos_windows`.  When
    ///   all per-negative weights are 1.0 (or `None`), this reduces to the raw
    ///   `n_neg_windows / n_pos_windows` ratio (mahbot-902).  The dynamic
    ///   formula (mahbot-993) prevents gradient imbalance when per-tier weights
    ///   (Ambient=1×, Owner=3×, Unrelated=10×, Confusable=15×) would otherwise
    ///   drown out the positive signal by ~13×.
    /// * `threshold` — Decision threshold (defaults to
    ///   [`DEFAULT_VERIFIER_THRESHOLD`] in production).
    /// * `l2_lambda` — L2 regularisation strength.
    /// * `rng_seed` — Optional seed for deterministic training (same seed +
    ///   same data = identical weights).  Production uses `None` (entropy-based).
    ///
    /// Returns a trained `VoiceVerifier`, or an untrained verifier if either
    /// input list is empty or no windows can be formed (all sequences shorter
    /// than [`VERIFIER_WINDOW_SIZE`] frames).
    ///
    /// ## Architecture
    ///
    /// Since mahbot-995, training produces a [`VerifierArch::Conv1D`] model
    /// (~581 params). Old [`VerifierArch::Logistic`] models remain loadable
    /// and functional via backward-compatible deserialization.
    #[must_use]
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn train(
        positive_sequences: &[EmbeddingSequence],
        negative_sequences: &[EmbeddingSequence],
        per_negative_sequence_weights: Option<&[f32]>,
        threshold: f32,
        l2_lambda: f32,
        rng_seed: Option<u64>,
    ) -> Self {
        Self::train_conv1d_inner(
            positive_sequences,
            negative_sequences,
            per_negative_sequence_weights,
            threshold,
            l2_lambda,
            rng_seed,
        )
    }

    /// Internal training: logistic regression on mean-pooled 96-dim windows.
    ///
    /// Only used by backward-compatibility tests (mahbot-995).
    #[cfg(test)]
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::similar_names,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn train_logistic_inner(
        positive_sequences: &[EmbeddingSequence],
        negative_sequences: &[EmbeddingSequence],
        per_negative_sequence_weights: Option<&[f32]>,
        threshold: f32,
        l2_lambda: f32,
        learning_rate: f32,
        max_iter: usize,
        rng_seed: Option<u64>,
    ) -> Self {
        // ── Prepare training data via shared helper ──
        let Some((
            mut windows,
            window_labels,
            window_weights,
            seq_infos,
            class_weight,
            _n_pos_windows,
        )) = prepare_training_data(
            positive_sequences,
            negative_sequences,
            per_negative_sequence_weights,
            form_sequence_windows,
        )
        else {
            return Self::untrained();
        };

        // If pre-windowed 288-dim input was provided, mean-pool to 96-dim
        // (logistic path, mahbot-901).  All windows formed by
        // form_sequence_windows are already 96-dim for per-frame input, but
        // pre-windowed 288-dim input needs pooling.
        if !windows.is_empty() && windows[0].len() != EMBEDDING_DIM {
            for w in &mut windows {
                let mut pooled = vec![0.0f32; EMBEDDING_DIM];
                mean_pool_window_into(w, &mut pooled);
                *w = pooled;
            }
            // Re-L2-normalize after pooling.
            for w in &mut windows {
                let norm = w.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
                for v in w.iter_mut() {
                    *v /= norm;
                }
            }
        }

        // ── Train/validation split (shared helper, source-tier stratified) ──
        let mut rng_split: StdRng = if let Some(seed) = rng_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(rand::random())
        };
        let (
            tr_windows,
            tr_labels,
            tr_weights,
            val_windows,
            val_labels,
            val_weights,
            used_sequence_split,
        ) = split_train_val(
            &seq_infos,
            &windows,
            &window_labels,
            &window_weights,
            &mut rng_split,
            true, // stratify_by_source — preserve negative tier proportions
        );

        // ── Train with validation, early stopping, and LR schedule ──
        // No StandardScaler (mahbot-996): the scaler is mathematically redundant
        // on L2-normalized embeddings. Logistic regression σ(w·x + b) can
        // represent any decision boundary that σ(w·(x-μ)/σ + b) can represent
        // because the affine transform (μ, σ) can be absorbed into the weights
        // and bias. Removing the scaler eliminates the OOD score-underflow
        // vulnerability where near-zero std dimensions produce extreme z-scores
        // that cause sigmoid underflow to exactly 0.0.
        let (weights, bias) = train_logistic_sgd_with_val(
            &tr_windows,
            &tr_labels,
            &tr_weights,
            &val_windows,
            &val_labels,
            &val_weights,
            l2_lambda,
            learning_rate,
            max_iter,
            EARLY_STOP_PATIENCE,
            rng_seed,
        );
        let verifier = Self {
            arch: VerifierArch::Logistic,
            trained: true,
            weights,
            bias,
            scaler_mean: Vec::new(),
            scaler_std: Vec::new(),
            conv_weight: Vec::new(),
            conv_bias: Vec::new(),
            fc_weight: Vec::new(),
            fc_bias: Vec::new(),
            threshold,
        };

        // Diagnostics: log training statistics and check discrimination.
        Self::log_training_diagnostics(
            &verifier,
            &tr_windows,
            &tr_labels,
            &val_windows,
            &val_labels,
            used_sequence_split,
            class_weight,
            l2_lambda,
        );

        verifier
    }

    /// Conv1D training path (mahbot-995).
    ///
    /// Trains a Conv1D(96→2, k=3, padding=1) → ReLU → AdaptiveAvgPool → Linear(2→1) → Sigmoid
    /// architecture using pure-Rust manual backprop + Adam.
    ///
    /// Reuses the shared infrastructure from [`train_logistic_inner`] for window
    /// formation, class weight calculation, and stratified per-sequence train/val
    /// split, but uses 288-dim concatenated windows (no mean-pooling) and
    /// Conv1D backprop (no StandardScaler).
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::similar_names,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn train_conv1d_inner(
        positive_sequences: &[EmbeddingSequence],
        negative_sequences: &[EmbeddingSequence],
        per_negative_sequence_weights: Option<&[f32]>,
        threshold: f32,
        l2_lambda: f32,
        rng_seed: Option<u64>,
    ) -> Self {
        // ── Prepare training data via shared helper ──
        let Some((windows, window_labels, window_weights, seq_infos, class_weight, _n_pos_windows)) =
            prepare_training_data(
                positive_sequences,
                negative_sequences,
                per_negative_sequence_weights,
                form_conv1d_sequence_windows,
            )
        else {
            return Self::untrained();
        };

        // All windows from form_conv1d_sequence_windows are already 288-dim
        // and L2-normalized — no mean-pooling needed.

        // ── Stratified per-sequence train/val split (shared helper) ──
        // Uses source-tier stratification matching the logistic path
        // (mahbot-949) for consistent training regimes between architectures.
        //
        // Create RNG here so it can be reused by the training loop below
        // for epoch-level shuffling (preserving deterministic seed behavior).
        let mut rng: StdRng = if let Some(seed) = rng_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(rand::random())
        };
        let (
            tr_windows,
            tr_labels,
            tr_weights,
            val_windows,
            val_labels,
            val_weights,
            used_sequence_split,
        ) = split_train_val(
            &seq_infos,
            &windows,
            &window_labels,
            &window_weights,
            &mut rng,
            true, // stratify_by_source — preserve negative tier proportions
        );

        // ── Conv1D training with Adam ──
        // Architecture: Conv1D(96→2, k=3, padding=1) → ReLU → AdaptiveAvgPool → Linear(2→1)
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

                    // ReLU
                    let mut relu_out = conv_out.clone();
                    crate::audio::wake_word_classifier::relu(&mut relu_out);
                    // Save mask for backward
                    let relu_mask: Vec<f32> = conv_out
                        .iter()
                        .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
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

                    // ReLU backward
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

                // L2 regularization (bias not regularized).
                for (g, w) in g_conv_w.iter_mut().zip(weight_conv.iter()) {
                    *g += l2_lambda * w;
                }
                for (g, w) in g_fc_w.iter_mut().zip(weight_fc.iter()) {
                    *g += l2_lambda * w;
                }

                // Adam step.
                opt_conv_w.update(&mut weight_conv, &g_conv_w, CONV_LEARNING_RATE);
                opt_conv_b.update(&mut bias_conv, &g_conv_b, CONV_LEARNING_RATE);
                opt_fc_w.update(&mut weight_fc, &g_fc_w, CONV_LEARNING_RATE);
                opt_fc_b.update(&mut bias_fc, &g_fc_b, CONV_LEARNING_RATE);
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
        let verifier = Self {
            arch: VerifierArch::Conv1D,
            weights: Vec::new(), // logistic fields unused
            bias: 0.0,
            scaler_mean: Vec::new(),
            scaler_std: Vec::new(),
            conv_weight: weight_conv,
            conv_bias: bias_conv,
            fc_weight: weight_fc,
            fc_bias: bias_fc,
            threshold,
            trained: true,
        };

        // Log diagnostics.
        log_verifier_diagnostics(
            &verifier,
            &tr_windows,
            &tr_labels,
            &val_windows,
            &val_labels,
            used_sequence_split,
            class_weight,
            "Conv1D verifier",
        );

        verifier
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

    /// Log post-training diagnostics and check for discrimination collapse.
    ///
    /// Logistic training diagnostics (backward-compat only, mahbot-995).
    /// Delegates to [`log_verifier_diagnostics`] for the shared logic.
    #[cfg(test)]
    #[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
    fn log_training_diagnostics(
        verifier: &Self,
        tr_windows: &[Vec<f32>],
        tr_labels: &[f32],
        val_windows: &[Vec<f32>],
        val_labels: &[f32],
        used_sequence_split: bool,
        class_weight: f32,
        _l2_lambda: f32, // logged by caller, not passed through
    ) {
        log_verifier_diagnostics(
            verifier,
            tr_windows,
            tr_labels,
            val_windows,
            val_labels,
            used_sequence_split,
            class_weight,
            "Verifier",
        );
    }

    /// Convenience: train a verifier using the given positive embeddings and
    /// automatically generated synthetic negative examples (distribution-
    /// matched via [`generate_synthetic_negatives_from_positives`] instead of
    /// pure N(0,1) Gaussian noise).
    ///
    /// Uses Conv1D training hyperparameters (mahbot-995).
    ///
    /// When `rng_seed` is `Some(seed)`, uses a seeded RNG for all random
    /// operations (synthetic negative generation + weight initialization),
    /// making training deterministic.  When `None`, uses entropy-based RNG
    /// (production path).
    #[must_use]
    pub fn train_with_synthetic_negatives(
        positive_sequences: &[EmbeddingSequence],
        threshold: f32,
        rng_seed: Option<u64>,
    ) -> Self {
        // Extract flat embeddings from all positive sequences for the helper.
        let flat_positives: Vec<Vec<f32>> = positive_sequences
            .iter()
            .flat_map(|s| s.embeddings.iter().cloned())
            .collect();
        let negatives = generate_synthetic_negatives_from_positives(
            SYNTHETIC_NEGATIVES_COUNT,
            &flat_positives,
            1.5, // noise_scale — matched to benchmark default
            rng_seed,
        );
        let synth_seq = EmbeddingSequence::negative(
            crate::audio::embedding_sequence::UtteranceId {
                sequence_index: 0,
                variant_index: 0,
            },
            crate::audio::embedding_sequence::Source::Synthetic,
            None,
            negatives,
        );
        Self::train(
            positive_sequences,
            &[synth_seq],
            None, // no per-negative weights for synthetic negatives
            threshold,
            CONV_L2_LAMBDA, // use Conv1D default L2 for synthetic bootstrapping
            rng_seed,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Shared training data preparation
// ═══════════════════════════════════════════════════════════════════════════

/// Shared training data preparation: forms windows, tracks [`SeqInfo`], computes
/// class weights, and L2-normalizes.
///
/// This eliminates ~70 lines of duplicated boilerplate between the logistic and
/// Conv1D training paths (mahbot-995).  The caller provides a `window_fn` that
/// transforms per-frame embeddings into windows — [`form_sequence_windows`] for
/// the logistic path (mean-pooled 96-dim) or [`form_conv1d_sequence_windows`]
/// for the Conv1D path (concatenated 288-dim).
///
/// # Returns
///
/// `Some((windows, labels, weights, seq_infos, class_weight, n_pos_windows))`
/// where all windows are L2-normalized.  Returns `None` and emits a `warn!`
/// log if no windows could be formed (all sequences shorter than
/// [`VERIFIER_WINDOW_SIZE`] frames or empty inputs).
#[allow(clippy::type_complexity, clippy::cast_precision_loss)]
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
            });
        }
    }
    let n_pos_windows = window_labels.iter().filter(|&&l| l > 0.5).count();

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
            });
        }
    }

    if windows.is_empty() {
        warn!(
            "Cannot form windows: need at least {VERIFIER_WINDOW_SIZE} per-frame embeddings per sequence",
        );
        return None;
    }

    // ── Class weight from window counts (mahbot-993) ──
    let class_weight = {
        let n_pw_f = n_pos_windows as f32;
        if n_pw_f > 0.0 {
            let neg_sum: f32 = window_weights[n_pos_windows..].iter().sum();
            neg_sum / n_pw_f
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

/// Shared per-sequence train/val split with optional source-tier stratification.
///
/// Accepts an existing `rng` (which the caller should seed for determinism) and
/// splits sequences by class (with optional stratification by [`SeqInfo::source`]),
/// shuffles and splits each group by [`VALIDATION_SPLIT`], gathers windows via
/// [`VoiceVerifier::gather_windows`], and falls back to a per-window random split
/// if the per-sequence split produces insufficient validation data.
///
/// When `stratify_by_source` is `true`, negative sequences are grouped by source
/// tier before splitting — preserving tier proportions in both train and val
/// sets (matches logistic verifier training, mahbot-949).
///
/// After the split, the caller can continue to use `rng` for subsequent
/// operations (e.g. epoch-level shuffling in the Conv1D training loop).
/// The logistic path's train/val split and the Conv1D path's train/val split
/// now share this single implementation (mahbot-995).
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
    bool,
) {
    let (tr_seq_idx, val_seq_idx) = if stratify_by_source {
        // ── Source-tier stratified split (logistic path, mahbot-949) ──
        // Group sequence indices by class and source tier, then shuffle
        // and assign to train/val stratified by group.
        let mut pos_indices: Vec<usize> = Vec::new();
        let mut neg_sources: std::collections::HashMap<
            crate::audio::embedding_sequence::Source,
            Vec<usize>,
        > = std::collections::HashMap::new();
        for (i, info) in seq_infos.iter().enumerate() {
            if info.is_positive {
                pos_indices.push(i);
            } else {
                neg_sources.entry(info.source).or_default().push(i);
            }
        }

        let mut tr_seq_idx: Vec<usize> = Vec::new();
        let mut val_seq_idx: Vec<usize> = Vec::new();

        pos_indices.shuffle(rng);
        let n_pos_train = ((pos_indices.len() as f32) * (1.0 - VALIDATION_SPLIT)).ceil() as usize;
        for (j, &idx) in pos_indices.iter().enumerate() {
            if j < n_pos_train {
                tr_seq_idx.push(idx);
            } else {
                val_seq_idx.push(idx);
            }
        }

        for indices in neg_sources.values() {
            let mut shuffled = indices.clone();
            shuffled.shuffle(rng);
            let n_train = ((shuffled.len() as f32) * (1.0 - VALIDATION_SPLIT)).ceil() as usize;
            for (j, &idx) in shuffled.iter().enumerate() {
                if j < n_train {
                    tr_seq_idx.push(idx);
                } else {
                    val_seq_idx.push(idx);
                }
            }
        }

        (tr_seq_idx, val_seq_idx)
    } else {
        // ── Simple pos/neg split (original Conv1D path, mahbot-995) ──
        let pos_seq_idx: Vec<usize> = seq_infos
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_positive)
            .map(|(i, _)| i)
            .collect();
        let neg_seq_idx: Vec<usize> = seq_infos
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.is_positive)
            .map(|(i, _)| i)
            .collect();

        let n_val_pos = ((pos_seq_idx.len() as f32) * VALIDATION_SPLIT).round() as usize;
        let n_val_neg = ((neg_seq_idx.len() as f32) * VALIDATION_SPLIT).round() as usize;

        let mut pos_shuffled = pos_seq_idx.clone();
        pos_shuffled.shuffle(rng);
        let mut neg_shuffled = neg_seq_idx.clone();
        neg_shuffled.shuffle(rng);

        let tr_seq_idx: Vec<usize> = pos_shuffled[n_val_pos.min(pos_shuffled.len())..]
            .iter()
            .chain(neg_shuffled[n_val_neg.min(neg_shuffled.len())..].iter())
            .copied()
            .collect();
        let val_seq_idx: Vec<usize> = pos_shuffled[..n_val_pos.min(pos_shuffled.len())]
            .iter()
            .chain(neg_shuffled[..n_val_neg.min(neg_shuffled.len())].iter())
            .copied()
            .collect();

        (tr_seq_idx, val_seq_idx)
    };

    // Build train/val arrays from split sequence indices.
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

    // If per-sequence split produced insufficient validation data, fall
    // back to per-window random split (matching Conv1D classifier behavior).
    if val.0.is_empty() || tr.0.len() < 2 {
        let total = windows.len();
        let n_val = ((total as f32) * VALIDATION_SPLIT).ceil() as usize;
        let n_val = n_val.max(1).min(total - 1);
        let mut idx: Vec<usize> = (0..total).collect();
        idx.shuffle(rng);
        let mut tr_f = Vec::with_capacity(total - n_val);
        let mut tr_l = Vec::with_capacity(total - n_val);
        let mut tr_w = Vec::with_capacity(total - n_val);
        let mut va_f = Vec::with_capacity(n_val);
        let mut va_l = Vec::with_capacity(n_val);
        let mut va_w = Vec::with_capacity(n_val);
        for (j, &i) in idx.iter().enumerate() {
            if j < n_val {
                va_f.push(windows[i].clone());
                va_l.push(window_labels[i]);
                va_w.push(window_weights[i]);
            } else {
                tr_f.push(windows[i].clone());
                tr_l.push(window_labels[i]);
                tr_w.push(window_weights[i]);
            }
        }
        (tr_f, tr_l, tr_w, va_f, va_l, va_w, false)
    } else {
        (tr.0, tr.1, tr.2, val.0, val.1, val.2, true)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Window helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Form windows from a per-frame embedding list (logistic path, mahbot-901).
///
/// Always uses mean-pooled 96-dim windows.
/// Only used by backward-compat tests (mahbot-995).
#[cfg(test)]
fn form_sequence_windows(embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if embeddings.is_empty() {
        return Vec::new();
    }
    if embeddings[0].len() == EMBEDDING_DIM {
        // Per-frame: form stride-1 mean-pooled windows.
        form_stride1_pooled_windows(embeddings)
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

/// Form windows from a per-frame embedding list for Conv1D training (mahbot-995).
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

/// Mean-pool three 96-dim embedding vectors into a 96-dim pooled vector.
///
/// Used by both the inference hot-path ([`mean_pool_window_into`]) and
/// training windowing ([`form_stride1_pooled_windows`]) to avoid duplicating
/// the averaging logic.
#[inline]
#[allow(clippy::cast_precision_loss)]
fn mean_pool_triple_into(frame0: &[f32], frame1: &[f32], frame2: &[f32], out: &mut [f32]) {
    for i in 0..EMBEDDING_DIM {
        out[i] = (frame0[i] + frame1[i] + frame2[i]) / VERIFIER_WINDOW_SIZE as f32;
    }
}

/// Mean-pool a 288-dim concatenated 3-frame window into a 96-dim pooled vector.
///
/// Writes into a stack-allocated `[f32; EMBEDDING_DIM]` buffer to avoid heap
/// allocation on the streaming inference hot path (mahbot-874).
///
/// # Panics
///
/// Panics if `window.len() != VERIFIER_INPUT_DIM` or `out.len() != EMBEDDING_DIM`.
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
    mean_pool_triple_into(f0, f1, f2, out);
}

/// Shared stride-1 window iteration primitive.
///
/// Extracts the common outer-loop scaffolding from [`form_stride1_windows`] and
/// [`form_stride1_pooled_windows`]: bounds check, capacity calculation, stride-1
/// iteration, L2-normalization, and push.  The caller provides a `form_window`
/// closure that fills a pre-allocated `window_size`-element buffer for each
/// window index `i`.
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

/// Form stride-1 mean-pooled windows from a flat list of 96-dim embeddings.
///
/// Each window is 3 consecutive embeddings mean-pooled into a 96-dim vector,
/// then L2-normalized.  Consecutive windows overlap by 2 embeddings (stride 1).
///
/// This is the logistic verifier counterpart of [`form_stride1_windows`]
/// (mahbot-901).  Only used by backward-compat tests (mahbot-995).
///
/// Returns empty vec if fewer than 3 embeddings are available.
#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
fn form_stride1_pooled_windows(embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
    stride1_windows_impl(embeddings, EMBEDDING_DIM, |i, out| {
        mean_pool_triple_into(&embeddings[i], &embeddings[i + 1], &embeddings[i + 2], out);
    })
}

/// Form stride-1 **concatenated** windows from a flat list of 96-dim embeddings.
///
/// Each window is 3 consecutive embeddings concatenated into a 288-dim vector,
/// then L2-normalized.  Consecutive windows overlap by 2 embeddings (stride 1).
///
/// This is the Conv1D verifier counterpart of the mean-pooled variant
/// ([`form_stride1_pooled_windows`]).  Instead of mean-pooling to 96-dim, it
/// preserves the full 288-dim temporal structure for the Conv1D layers
/// (mahbot-995).
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

/// Standard sigmoid function: `1 / (1 + e^{-x})`.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Logistic regression inference path for mean-pooled 96-dim verifier (mahbot-901).
///
/// Pipeline: 288-dim input → mean-pool to 96-dim → L2-normalize → StandardScaler
/// → dot(weights, scaled) + bias → sigmoid.
///
/// Accepts either 288-dim input (mean-pools first) or 96-dim already-pooled
/// input (skips pooling, e.g. from training diagnostics).
///
/// All intermediate buffers are stack-allocated (96 f32s = 384 bytes each) to
/// avoid heap allocation on the streaming inference hot path (mahbot-874).
///
/// # Panics
///
/// Panics if the input dimension is neither 288 (needs pooling) nor 96 (already pooled).
fn predict_logistic(
    embedding: &[f32],
    weights: &[f32],
    bias: f32,
    scaler_mean: &[f32],
    scaler_std: &[f32],
) -> f32 {
    // Step 1: If 288-dim input, mean-pool to 96-dim.  If already 96-dim, use directly.
    let pooled: [f32; EMBEDDING_DIM] = if embedding.len() == VERIFIER_INPUT_DIM {
        let mut p = [0.0f32; EMBEDDING_DIM];
        mean_pool_window_into(embedding, &mut p);
        p
    } else {
        assert_eq!(
            embedding.len(),
            EMBEDDING_DIM,
            "Logistic verifier expects {VERIFIER_INPUT_DIM}-dim or {EMBEDDING_DIM}-dim input, got {}",
            embedding.len(),
        );
        let mut p = [0.0f32; EMBEDDING_DIM];
        p.copy_from_slice(embedding);
        p
    };

    // Step 2: L2-normalize the pooled 96-dim vector (unit-sphere projection).
    let norm_l2: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
    let mut x_l2 = [0.0f32; EMBEDDING_DIM];
    #[allow(clippy::cast_precision_loss)]
    for (i, &v) in pooled.iter().enumerate() {
        x_l2[i] = v / norm_l2;
    }

    // Step 3: Apply StandardScaler on the L2-normalized 96-dim values.
    let use_scaler = !scaler_mean.is_empty() && !scaler_std.is_empty();
    let mut x = [0.0f32; EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        x[i] = if use_scaler && scaler_std[i] > 0.0 {
            (x_l2[i] - scaler_mean[i]) / scaler_std[i]
        } else {
            x_l2[i]
        };
    }

    // Step 4: Linear combination z = w·x + b → sigmoid.
    let z: f32 = x
        .iter()
        .zip(weights.iter())
        .map(|(v, w)| v * w)
        .sum::<f32>()
        + bias;
    sigmoid(z)
}

/// Conv1D inference path for the verifier (mahbot-995).
///
/// Architecture: Conv1D(96→2, k=3, padding=1) → ReLU → AdaptiveAvgPool1d → Linear(2→1) → Sigmoid.
///
/// Input must be 288-dim (3 concatenated 96-dim embeddings). Panics otherwise.
///
/// Pipeline: 288-dim input → L2-normalize → Reshape to channels-first [96 × 3]
/// → Conv1D → ReLU → AdaptiveAvgPool1d → Linear → Sigmoid.
///
/// # Panics
///
/// Panics if `embedding.len() != VERIFIER_INPUT_DIM`.
#[allow(clippy::cast_precision_loss)]
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

    let cin = EMBEDDING_DIM; // 96
    let l = VERIFIER_WINDOW_SIZE; // 3
    let cout = CONV_VERIFIER_OUT; // 2
    let ks = CONV_VERIFIER_KERNEL_SIZE; // 3

    // Step 1: L2-normalize the 288-dim input (matching training pipeline).
    // Uses a stack-allocated [f32; 288] buffer to avoid heap allocation on the
    // streaming inference hot path (matching predict_logistic's pattern).
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
    let cf = crate::audio::wake_word_classifier::to_channels_first(&x, cin, l);

    // Step 3: Conv1D(96 → 2, k=3, padding=1).
    let mut conv_out =
        crate::audio::wake_word_classifier::conv1d(&cf, cin, l, cout, ks, conv_weight, conv_bias);

    // Step 4: ReLU activation.
    crate::audio::wake_word_classifier::relu(&mut conv_out);

    // Step 5: AdaptiveAvgPool1d (3 → 1) — average over the time dimension.
    let pooled = crate::audio::wake_word_classifier::adaptive_avg_pool(&conv_out, cout, l);

    // Step 6: Linear(2 → 1) → Sigmoid.
    let logit: f32 = pooled
        .iter()
        .zip(fc_weight.iter())
        .map(|(v, w)| v * w)
        .sum::<f32>()
        + fc_bias[0];
    1.0 / (1.0 + (-logit).exp())
}

/// Compute per-dimension mean and population standard deviation for
/// StandardScaler normalisation (testing only, mahbot-995).
///
/// Returns an empty `(Vec, Vec)` pair when `features` is empty.
#[cfg(test)]
fn compute_standard_scaler(features: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    if features.is_empty() || features[0].is_empty() {
        return (Vec::new(), Vec::new());
    }

    let dim = features[0].len();
    #[allow(clippy::cast_precision_loss)]
    let n = features.len() as f32;

    // ── Mean per dimension ──
    let mut mean = vec![0.0; dim];
    for feat in features {
        for (j, &val) in feat.iter().enumerate() {
            mean[j] += val;
        }
    }
    for m in &mut mean {
        *m /= n;
    }

    // ── Population std per dimension (ddof=0) ──
    let mut std = vec![0.0; dim];
    for feat in features {
        for (j, &val) in feat.iter().enumerate() {
            let diff = val - mean[j];
            std[j] += diff * diff;
        }
    }
    for s in &mut std {
        *s = (*s / n).sqrt().max(SCALER_STD_MIN);
    }

    (mean, std)
}

// ═══════════════════════════════════════════════════════════════════════════
// Logistic regression SGD with validation + early stopping
// ═══════════════════════════════════════════════════════════════════════════

/// Train a logistic regression classifier with validation-based early stopping
/// and LR step decay (testing only, mahbot-995).
///
/// Same architecture as [`train_logistic_sgd`] but accepts separate validation
/// data and implements:
/// - Early stopping: stops if validation loss hasn't improved for `patience` steps
/// - Loss logging: logs train and val loss every `LOG_LOSS_INTERVAL` iterations
/// - LR step decay: halves the learning rate unconditionally every 200 iterations
///
/// When `val_features` is empty, runs without early stopping (full `max_iter`
/// iterations), matching the original `train_logistic_sgd` behavior.
#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
fn train_logistic_sgd_with_val(
    features: &[Vec<f32>],     // scaled training features (n × 96)
    labels: &[f32],            // training labels: 0.0 or 1.0
    sample_weights: &[f32],    // per-sample weight (n)
    val_features: &[Vec<f32>], // scaled validation features (may be empty)
    val_labels: &[f32],        // validation labels
    val_weights: &[f32],       // validation per-sample weights
    l2_lambda: f32,
    learning_rate: f32,
    max_iter: usize,
    patience: usize,
    rng_seed: Option<u64>,
) -> (Vec<f32>, f32) {
    let n = features.len();
    if n == 0 {
        return (Vec::new(), 0.0);
    }
    let dim = features[0].len();
    if dim == 0 {
        return (Vec::new(), 0.0);
    }

    let n_f32 = n as f32;
    let use_val = !val_features.is_empty() && !val_labels.is_empty();

    // ── Initialize weights to small random values (bias starts at 0) ──
    let mut rng: StdRng = if let Some(seed) = rng_seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::seed_from_u64(rand::random())
    };

    // Xavier-like init for logistic: sqrt(1/dim) scale (Glorot for fan-in only).
    let init_scale = (1.0 / dim as f32).sqrt();
    let mut weights = vec![0.0; dim];
    for w in &mut weights {
        *w = rng.random::<f32>() * 2.0 * init_scale - init_scale;
    }
    let mut bias = 0.0;

    // ── Early stopping state ──
    let mut best_weights = weights.clone();
    let mut best_bias = bias;
    let mut best_loss = f32::INFINITY;
    let mut stall_count = 0;
    let mut current_lr = learning_rate;

    // ── SGD training loop ──
    for iter in 0..max_iter {
        // LR step decay: halve every 200 iterations (mahbot-949 Fix 5).
        if iter > 0 && iter % 200 == 0 {
            current_lr *= 0.5;
        }

        let mut dw = vec![0.0; dim];
        let mut db = 0.0;

        for i in 0..n {
            // Forward: z = w·x + b → sigmoid
            let mut z = bias;
            for d in 0..dim {
                z += weights[d] * features[i][d];
            }
            let pred = sigmoid(z);
            let y = labels[i];
            let w_i = sample_weights[i];

            // Gradient of binary cross-entropy (weighted):
            // dL/dz = w_i * (pred - y)
            let dz = w_i * (pred - y);

            // dL/dw_d = dz * x_d
            for d in 0..dim {
                dw[d] += dz * features[i][d];
            }
            db += dz;
        }

        // Average gradients over batch.
        for d in &mut dw {
            *d /= n_f32;
        }
        db /= n_f32;

        // Add L2 regularization gradient: λ * w (bias not regularized).
        for d in 0..dim {
            dw[d] += l2_lambda * weights[d];
        }

        for d in 0..dim {
            weights[d] -= current_lr * dw[d];
        }
        bias -= current_lr * db;

        // ── Validation loss + early stopping ──
        if use_val {
            let val_loss = compute_weighted_bce_loss(
                val_features,
                val_labels,
                val_weights,
                &weights,
                bias,
                l2_lambda,
                false, // validation loss excludes L2 term (mahbot-949)
            );
            let train_loss = compute_weighted_bce_loss(
                features,
                labels,
                sample_weights,
                &weights,
                bias,
                l2_lambda,
                true, // training loss includes L2 for monitoring consistency
            );

            if iter % LOG_LOSS_INTERVAL == 0 {
                info!(
                    "Verifier SGD: iter={iter} train_loss={train_loss:.6} val_loss={val_loss:.6} lr={current_lr:.6}",
                );
            }

            if val_loss < best_loss - 1e-8 {
                best_loss = val_loss;
                best_weights.clone_from(&weights);
                best_bias = bias;
                stall_count = 0;
            } else {
                stall_count += 1;
                if stall_count >= patience {
                    info!(
                        "Verifier SGD early stop at iter={iter}: best_val_loss={best_loss:.6} (patience={patience})",
                    );
                    weights.copy_from_slice(&best_weights);
                    bias = best_bias;
                    break;
                }
            }
        } else if iter % LOG_LOSS_INTERVAL == 0 {
            let train_loss = compute_weighted_bce_loss(
                features,
                labels,
                sample_weights,
                &weights,
                bias,
                l2_lambda,
                true, // training loss includes L2 for monitoring consistency
            );
            info!("Verifier SGD: iter={iter} train_loss={train_loss:.6} lr={current_lr:.6}",);
        }
    }

    // If early stopping never triggered and we used validation, restore best.
    if use_val && stall_count < patience {
        weights.copy_from_slice(&best_weights);
        bias = best_bias;
    }

    (weights, bias)
}

/// Compute the weighted binary cross-entropy loss (testing only, mahbot-995).
#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
fn compute_weighted_bce_loss(
    features: &[Vec<f32>],
    labels: &[f32],
    sample_weights: &[f32],
    weights: &[f32],
    bias: f32,
    l2_lambda: f32,
    use_l2: bool,
) -> f32 {
    let n = features.len();
    if n == 0 {
        return 0.0;
    }
    let mut total = 0.0f32;
    for i in 0..n {
        let mut z = bias;
        for d in 0..weights.len() {
            z += weights[d] * features[i][d];
        }
        let pred = sigmoid(z);
        let eps = 1e-10;
        total += sample_weights[i]
            * (labels[i] * (pred + eps).ln() + (1.0 - labels[i]) * (1.0 - pred + eps).ln());
    }
    let bce = -total / n as f32;
    if use_l2 {
        // Add L2 regularization term: (λ/2) * ||w||²
        let l2_term = 0.5 * l2_lambda * weights.iter().map(|w| w * w).sum::<f32>();
        bce + l2_term
    } else {
        bce
    }
}

/// Compute weighted binary cross-entropy loss for Conv1D verifier (mahbot-995).
///
/// Runs a forward pass through the Conv1D architecture for each sample.
/// When `use_l2` is true, includes L2 regularization on conv_weight and fc_weight.
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
        crate::audio::wake_word_classifier::relu(&mut relu_out);
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
                + fc_weight.iter().map(|w| w * w).sum::<f32>());
        bce + l2_term
    } else {
        bce
    }
}

/// Log verifier training diagnostics and check for discrimination collapse.
///
/// Shared between Conv1D (production) and logistic (backward-compat test) paths.
/// The `label` parameter controls the log prefix (e.g. "Conv1D verifier", "Verifier").
#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
fn log_verifier_diagnostics(
    verifier: &VoiceVerifier,
    tr_windows: &[Vec<f32>],
    tr_labels: &[f32],
    val_windows: &[Vec<f32>],
    val_labels: &[f32],
    used_sequence_split: bool,
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

    let split_method = if used_sequence_split {
        "per-sequence"
    } else {
        "per-window"
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
// Logistic regression SGD (original, no validation)
// ═══════════════════════════════════════════════════════════════════════════

/// Train a logistic regression classifier on scaled 96-dim features using SGD.
/// Only used by backward-compat tests (mahbot-995).
#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
fn train_logistic_sgd(
    features: &[Vec<f32>],  // scaled (n × 96)
    labels: &[f32],         // 0.0 or 1.0
    sample_weights: &[f32], // per-sample weight (n)
    l2_lambda: f32,
    learning_rate: f32,
    max_iter: usize,
    rng_seed: Option<u64>, // deterministic training when Some
) -> (Vec<f32>, f32) {
    let n = features.len();
    if n == 0 {
        return (Vec::new(), 0.0);
    }
    let dim = features[0].len();
    if dim == 0 {
        return (Vec::new(), 0.0);
    }

    let n_f32 = n as f32;

    // ── Initialize weights to small random values (bias starts at 0) ──
    let mut rng: StdRng = if let Some(seed) = rng_seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::seed_from_u64(rand::random())
    };

    // Xavier-like init for logistic: sqrt(1/dim) scale (Glorot for fan-in only).
    let init_scale = (1.0 / dim as f32).sqrt();
    let mut weights = vec![0.0; dim];
    for w in &mut weights {
        *w = rng.random::<f32>() * 2.0 * init_scale - init_scale;
    }
    let mut bias = 0.0;

    // ── SGD training loop ──
    for _iter in 0..max_iter {
        let mut dw = vec![0.0; dim];
        let mut db = 0.0;

        for i in 0..n {
            // Forward: z = w·x + b → sigmoid
            let mut z = bias;
            for d in 0..dim {
                z += weights[d] * features[i][d];
            }
            let pred = sigmoid(z);
            let y = labels[i];
            let w_i = sample_weights[i];

            // Gradient of binary cross-entropy (weighted):
            // dL/dz = w_i * (pred - y)
            let dz = w_i * (pred - y);

            // dL/dw_d = dz * x_d
            for d in 0..dim {
                dw[d] += dz * features[i][d];
            }
            db += dz;
        }

        // Average gradients over batch.
        for d in &mut dw {
            *d /= n_f32;
        }
        db /= n_f32;

        // Add L2 regularization gradient: λ * w (bias not regularized).
        for d in 0..dim {
            dw[d] += l2_lambda * weights[d];
        }

        for d in 0..dim {
            weights[d] -= learning_rate * dw[d];
        }
        bias -= learning_rate * db;
    }

    (weights, bias)
}
// ═══════════════════════════════════════════════════════════════════════════
// Synthetic negatives
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// Synthetic negatives
// ═══════════════════════════════════════════════════════════════════════════

/// Generate `count` synthetic negative embeddings of dimension `dim` using
/// Gaussian noise (Box-Muller transform).
///
/// Each embedding is drawn from N(0, 1), which approximates the distribution
/// of normalised real embeddings. This provides a weak but useful
/// bootstrapping signal for the verifier when real calibration negatives are
/// not yet available.
#[cfg(test)]
#[must_use]
pub(crate) fn generate_synthetic_negatives(count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    // Box-Muller transform: generate N(0,1) from two
                    // independent uniforms in (0, 1].
                    loop {
                        let u1: f32 = rand::random();
                        let u2: f32 = rand::random();
                        // Guard: avoid ln(0) = -inf.  Both must be strictly
                        // positive to avoid degenerate samples.
                        if u1 > 0.0 && u2 > 0.0 {
                            let r = (-2.0 * u1.ln()).sqrt();
                            let theta = 2.0 * std::f32::consts::PI * u2;
                            break r * theta.cos();
                        }
                    }
                })
                .collect()
        })
        .collect()
}

/// Generate synthetic negative embeddings based on the statistics of the
/// positive embeddings (mahbot-846).  Unlike [`generate_synthetic_negatives`]
/// which produces pure N(0,1) noise in a completely different region of
/// embedding space than real speech, this function produces negatives that
/// overlap with the real embedding distribution.
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
                    // Box-Muller N(0,1)
                    let z = loop {
                        let u1: f32 = rng.random();
                        let u2: f32 = rng.random();
                        if u1 > 0.0 && u2 > 0.0 {
                            let r = (-2.0 * u1.ln()).sqrt();
                            let theta = 2.0 * std::f32::consts::PI * u2;
                            break r * theta.cos();
                        }
                    };
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
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Helper: wrap flat embeddings into a single EmbeddingSequence for testing.
    fn make_seq(
        embs: Vec<Vec<f32>>,
        label: crate::audio::embedding_sequence::LabelStratum,
    ) -> EmbeddingSequence {
        EmbeddingSequence {
            id: crate::audio::embedding_sequence::UtteranceId {
                sequence_index: 0,
                variant_index: 0,
            },
            source: crate::audio::embedding_sequence::Source::Enrollment,
            augmentation_family: None,
            label_stratum: label,
            embeddings: embs,
        }
    }

    /// Generate a synthetic 288-dim "positive" window with values clustered
    /// around +0.5 (simulating a wake-word embedding window).
    fn make_positive_embedding(rng: &mut impl Rng) -> Vec<f32> {
        (0..VERIFIER_INPUT_DIM)
            .map(|_| {
                // Positive cluster: N(0.5, 0.3)
                loop {
                    let u1: f32 = rng.random();
                    let u2: f32 = rng.random();
                    if u1 > 0.0 && u2 > 0.0 {
                        let r = (-2.0 * u1.ln()).sqrt();
                        let theta = 2.0 * std::f32::consts::PI * u2;
                        break 0.5 + 0.3 * r * theta.cos();
                    }
                }
            })
            .collect()
    }

    /// Generate a synthetic 288-dim "negative" window with values clustered
    /// around -0.5 (simulating a non-wake-word embedding window).
    fn make_negative_embedding(rng: &mut impl Rng) -> Vec<f32> {
        (0..VERIFIER_INPUT_DIM)
            .map(|_| {
                // Negative cluster: N(-0.5, 0.3)
                loop {
                    let u1: f32 = rng.random();
                    let u2: f32 = rng.random();
                    if u1 > 0.0 && u2 > 0.0 {
                        let r = (-2.0 * u1.ln()).sqrt();
                        let theta = 2.0 * std::f32::consts::PI * u2;
                        break -0.5 + 0.3 * r * theta.cos();
                    }
                }
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
                loop {
                    let u1: f32 = rng.random();
                    let u2: f32 = rng.random();
                    if u1 > 0.0 && u2 > 0.0 {
                        let r = (-2.0 * u1.ln()).sqrt();
                        let theta = 2.0 * std::f32::consts::PI * u2;
                        break 0.0 + 0.6 * r * theta.cos();
                    }
                }
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
        assert_weight_tier(&empty_weights, 3, 0, 0.0, "empty-at-end");

        // Mismatch should panic with descriptive message — verify via catch_unwind.
        let mismatch_weights = vec![1.0, 1.0, 2.0];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_weight_tier(&mismatch_weights, 0, 3, 1.0, "first");
        }));
        assert!(
            result.is_err(),
            "assert_weight_tier should panic on mismatch"
        );
    }

    #[test]
    fn assert_weight_tier_values_within_epsilon_pass() {
        // Values within f32::EPSILON (inclusive) of expected should NOT panic.
        // This exercises the floating-point equality boundary: the function
        // uses `<= f32::EPSILON`, so a value exactly EPSILON away should pass.
        let weights = vec![1.0f32 + f32::EPSILON, 1.0f32 - f32::EPSILON];
        assert_weight_tier(&weights, 0, 2, 1.0, "epsilon-boundary");
    }

    // ── Required tests (from ticket mahbot-777) ─────────────────────

    #[test]
    fn test_verifier_accepts_positive_rejects_negative() {
        // Train on known positive and negative synthetic embeddings, then verify
        // both acceptance of held-out positives and rejection of held-out negatives
        // (consolidated from two separate tests with identical setup, mahbot-874).
        let mut rng = StdRng::seed_from_u64(42);
        let positives: Vec<Vec<f32>> = (0..20).map(|_| make_positive_embedding(&mut rng)).collect();
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
        assert_eq!(verifier.fc_bias.len(), 1, "fc_bias must be 1-dim",);
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
        let positives: Vec<Vec<f32>> = (0..20).map(|_| make_positive_embedding(&mut rng)).collect();
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
            L2_LAMBDA,                  // L2 regularization (disabled: λ=0.0 since mahbot-994)
            Some(42),                   // deterministic seed for reproducibility
        );

        assert!(verifier.is_trained(), "Verifier must be trained");

        // Verify discrimination: held-out positive > held-out negative.
        // Uses relative comparison because Conv1D produces scores on a
        // different scale than logistic regression (mahbot-995).
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
    fn test_train_with_synthetic_negatives_rejects_non_wake_word_speech() {
        // Tests the actual production fallback path (mahbot-797):
        // when fewer than 2 real negative chunks are available, the verifier
        // is trained via train_with_synthetic_negatives which generates
        // synthetic Gaussian N(0,1) negatives internally. This verifies that
        // the resulting decision boundary correctly rejects non-wake-word
        // speech embeddings (unlike the old pre-fix verifier which would
        // accept any speech because it was trained only on N(0,1) noise).
        let mut rng = StdRng::seed_from_u64(99);
        let positives: Vec<Vec<f32>> = (0..30).map(|_| make_positive_embedding(&mut rng)).collect();
        let pos_seq = make_seq(
            positives,
            crate::audio::embedding_sequence::LabelStratum::Positive,
        );

        // Generate synthetic negatives (same logic as train_with_synthetic_negatives).
        let flat_positives: Vec<Vec<f32>> = pos_seq.embeddings.iter().cloned().collect();
        let negs = generate_synthetic_negatives_from_positives(
            SYNTHETIC_NEGATIVES_COUNT,
            &flat_positives,
            1.5, // noise_scale
            Some(99),
        );
        let synth_seq = EmbeddingSequence::negative(
            crate::audio::embedding_sequence::UtteranceId {
                sequence_index: 0,
                variant_index: 0,
            },
            crate::audio::embedding_sequence::Source::Synthetic,
            None,
            negs,
        );

        // Use logistic training path (train_logistic_inner) because synthetic
        // Gaussian negatives lack temporal structure — Conv1D requires real
        // speech temporal patterns to learn effectively. The logistic path is
        // kept as a backward-compatible fallback for this specific case
        // (mahbot-995).
        let verifier = VoiceVerifier::train_logistic_inner(
            &[pos_seq],
            &[synth_seq],
            None, // no per-negative weights for synthetic negatives
            DEFAULT_VERIFIER_THRESHOLD,
            L2_LAMBDA,
            LOGISTIC_LEARNING_RATE,
            LOGISTIC_MAX_ITER,
            Some(99),
        );

        assert!(verifier.is_trained(), "Verifier must be trained");
        assert_eq!(
            verifier.threshold, DEFAULT_VERIFIER_THRESHOLD,
            "threshold must match DEFAULT_VERIFIER_THRESHOLD",
        );

        // Structural assertions: logistic weights dimension, scaler empty, finite.
        assert_eq!(
            verifier.weights.len(),
            EMBEDDING_DIM,
            "weights should be {EMBEDDING_DIM}-dim",
        );
        assert!(
            verifier.scaler_mean.is_empty(),
            "scaler_mean should be empty (mahbot-996: scaler removed from training)"
        );
        assert!(
            verifier.scaler_std.is_empty(),
            "scaler_std should be empty (mahbot-996: scaler removed from training)"
        );
        for (j, &w) in verifier.weights.iter().enumerate() {
            assert!(
                w.is_finite(),
                "weights[{j}] is not finite: {w}; gradient descent diverged",
            );
        }
        assert!(
            verifier.bias.is_finite(),
            "bias is not finite; gradient descent diverged",
        );

        // Verify a held-out positive is accepted.
        let held_out = make_positive_embedding(&mut rng);
        let score = verifier.predict(&held_out);
        assert!(
            score >= 0.5,
            "Verifier should accept positive embedding (score >= 0.5), got score={score:.4}",
        );

        // Verify a held-out non-wake-word speech embedding is rejected.
        let held_out = make_non_wake_speech_embedding(&mut rng);
        let score = verifier.predict(&held_out);
        assert!(
            score < 0.5,
            "Verifier should reject non-wake-word speech embedding (score < 0.5), \
             got score={score:.4}",
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
    fn test_logistic_verifier_serialization_roundtrip() {
        // Train a logistic model and verify JSON roundtrip preserves predictions
        // and is_trained() status.
        let mut rng = StdRng::seed_from_u64(42);
        let positives: Vec<Vec<f32>> = (0..30).map(|_| make_positive_frame(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..50).map(|_| make_negative_frame(&mut rng)).collect();

        // L2-normalize positives and negatives separately.
        let l2_normalize = |v: &[f32]| -> Vec<f32> {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            v.iter().map(|x| x / norm).collect()
        };
        let pos_norm: Vec<Vec<f32>> = positives.iter().map(|f| l2_normalize(f)).collect();
        let neg_norm: Vec<Vec<f32>> = negatives.iter().map(|f| l2_normalize(f)).collect();

        // Train logistic SGD directly on L2-normalized features (no StandardScaler,
        // mahbot-996). The scaler is mathematically redundant on L2-normalized
        // embeddings — removing it eliminates the OOD score-underflow vulnerability.
        let mut all_norm = pos_norm;
        all_norm.extend(neg_norm);

        let labels: Vec<f32> = [vec![1.0; 30], vec![0.0; 50]].concat();
        let sample_weights: Vec<f32> = [vec![3.0; 30], vec![1.0; 50]].concat();

        let (weights, bias) = train_logistic_sgd(
            &all_norm,
            &labels,
            &sample_weights,
            L2_LAMBDA,
            0.01,
            500,
            Some(42),
        );

        // Build a logistic verifier with empty scaler (mahbot-996).
        let verifier = VoiceVerifier {
            arch: VerifierArch::Logistic,
            weights,
            bias,
            scaler_mean: Vec::new(),
            scaler_std: Vec::new(),
            conv_weight: Vec::new(),
            conv_bias: Vec::new(),
            fc_weight: Vec::new(),
            fc_bias: Vec::new(),
            threshold: DEFAULT_VERIFIER_THRESHOLD,
            trained: true,
        };

        // Verify it's considered trained.
        assert!(verifier.is_trained());

        // Serialize to JSON.
        let json = serde_json::to_string(&verifier).expect("serialize");

        // Deserialize.
        let deserialized: VoiceVerifier = serde_json::from_str(&json).expect("deserialize");

        // Verify is_trained() works on deserialized model.
        assert!(
            deserialized.is_trained(),
            "deserialized logistic verifier should be trained",
        );

        // Verify predictions match on held-out test vectors.
        let held_out_pos = make_positive_frame(&mut rng);
        let held_out_neg = make_negative_frame(&mut rng);

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

    // ── Additional correctness tests ────────────────────────────────

    #[test]
    fn test_sigmoid_symmetry() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6, "sigmoid(0) != 0.5");
        assert!((sigmoid(10.0) - 1.0).abs() < 1e-4, "sigmoid(10) != ~1.0",);
        assert!((sigmoid(-10.0) - 0.0).abs() < 1e-4, "sigmoid(-10) != ~0.0",);
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
    fn test_generate_synthetic_negatives() {
        let negs = generate_synthetic_negatives(10, 96);
        assert_eq!(negs.len(), 10);
        assert_eq!(negs[0].len(), 96);
        // All values should be finite (no NaN or Inf from Box-Muller).
        for emb in &negs {
            for &v in emb {
                assert!(v.is_finite(), "Synthetic negative has non-finite value {v}");
            }
        }

        // Zero count returns empty.
        assert!(generate_synthetic_negatives(0, 96).is_empty());
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
    fn test_compute_standard_scaler_basic() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let (mean, std) = compute_standard_scaler(&features);
        assert!((mean[0] - 3.0).abs() < 1e-6);
        assert!((mean[1] - 4.0).abs() < 1e-6);
        // Population std: sqrt((4+0+4)/3) ≈ 1.63299
        assert!((std[0] - 1.632_99).abs() < 1e-4);
        assert!((std[1] - 1.632_99).abs() < 1e-4);

        // Empty input produces empty scaler.
        let (mean, std) = compute_standard_scaler(&[]);
        assert!(mean.is_empty());
        assert!(std.is_empty());

        // Zero-variance input: all values identical → std would be 0.0 without
        // clamping → must be clamped to SCALER_STD_MIN (mahbot-996).
        let constant = vec![vec![5.0; 96]; 10];
        let (mean, std) = compute_standard_scaler(&constant);
        assert!((mean[0] - 5.0).abs() < 1e-6);
        for &s in &std {
            assert!(
                (s - 1e-3).abs() < 1e-7,
                "Zero-variance dimension must be clamped to 1e-3, got {s}",
            );
        }
    }

    #[test]
    fn test_verifier_rejects_mismatched_scaler_dims() {
        // A verifier with trained=true but scaler dimensions that don't match
        // weights must be detected as untrained.
        let verifier = VoiceVerifier {
            arch: VerifierArch::Logistic,
            trained: true,
            weights: vec![0.5; EMBEDDING_DIM],
            bias: 0.0,
            scaler_mean: vec![0.1; 48], // wrong dimension (48 ≠ 96)
            scaler_std: vec![0.2; 48],
            conv_weight: Vec::new(),
            conv_bias: Vec::new(),
            fc_weight: Vec::new(),
            fc_bias: Vec::new(),
            threshold: DEFAULT_VERIFIER_THRESHOLD,
        };
        assert!(
            !verifier.is_trained(),
            "Mismatched scaler dims should report untrained"
        );

        // Also test partial mismatch: only scaler_std populated.
        let verifier2 = VoiceVerifier {
            arch: VerifierArch::Logistic,
            trained: true,
            weights: vec![0.5; EMBEDDING_DIM],
            bias: 0.0,
            scaler_mean: Vec::new(),
            scaler_std: vec![0.2; 48], // non-empty but mismatched
            conv_weight: Vec::new(),
            conv_bias: Vec::new(),
            fc_weight: Vec::new(),
            fc_bias: Vec::new(),
            threshold: DEFAULT_VERIFIER_THRESHOLD,
        };
        assert!(
            !verifier2.is_trained(),
            "Partial mismatched scaler should report untrained"
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
        let positives: Vec<Vec<f32>> = (0..10).map(|_| make_positive_embedding(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..10).map(|_| make_negative_embedding(&mut rng)).collect();
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
            L2_LAMBDA,
            Some(seed),
        );
        let v2 = VoiceVerifier::train(
            &[pos_seq],
            &[neg_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            L2_LAMBDA,
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

        // ── Low-level train_logistic_sgd prediction check ───────────
        let mut rng2 = StdRng::seed_from_u64(42);
        let pos_frames: Vec<Vec<f32>> = (0..30).map(|_| make_positive_frame(&mut rng2)).collect();
        let neg_frames: Vec<Vec<f32>> = (0..50).map(|_| make_negative_frame(&mut rng2)).collect();

        let features: Vec<Vec<f32>> = pos_frames
            .iter()
            .chain(neg_frames.iter())
            .cloned()
            .collect();
        let labels: Vec<f32> = [vec![1.0; 30], vec![0.0; 50]].concat();
        let sample_weights: Vec<f32> = [vec![50.0 / 30.0; 30], vec![1.0; 50]].concat();

        let (weights, bias) = train_logistic_sgd(
            &features,
            &labels,
            &sample_weights,
            L2_LAMBDA,
            LOGISTIC_LEARNING_RATE,
            LOGISTIC_MAX_ITER,
            Some(42),
        );

        assert_eq!(
            weights.len(),
            EMBEDDING_DIM,
            "logistic weights should be 96-dim"
        );
        assert!(bias.is_finite(), "bias must be finite");
        for (j, &w) in weights.iter().enumerate() {
            assert!(
                w.is_finite(),
                "weights[{j}] is not finite; training diverged"
            );
        }

        // With L2_LAMBDA=0.0 and easily separable synthetic data, weights must
        // stay within reasonable magnitudes (no divergence from unregularized SGD).
        for (j, &w) in weights.iter().enumerate() {
            assert!(
                w.abs() < 100.0,
                "weights[{j}] magnitude {w:.2} exceeds 100; unregularized SGD diverged",
            );
        }
        assert!(
            bias.abs() < 100.0,
            "bias magnitude {bias:.2} exceeds 100; unregularized SGD diverged",
        );

        // Predict on held-out frames and verify discrimination.
        let held_out_pos: Vec<f32> = make_positive_frame(&mut rng2);
        let held_out_neg: Vec<f32> = make_negative_frame(&mut rng2);

        let score_pos = predict_logistic(&held_out_pos, &weights, bias, &[], &[]);
        let score_neg = predict_logistic(&held_out_neg, &weights, bias, &[], &[]);
        assert!(
            score_pos > score_neg,
            "Logistic should score positive ({score_pos:.4}) higher than negative ({score_neg:.4})",
        );

        // ── Low-level train_logistic_sgd deterministic check ────────
        let mut rng3 = StdRng::seed_from_u64(12345);
        let pos_det: Vec<Vec<f32>> = (0..10).map(|_| make_positive_frame(&mut rng3)).collect();
        let neg_det: Vec<Vec<f32>> = (0..10).map(|_| make_negative_frame(&mut rng3)).collect();
        let det_features: Vec<Vec<f32>> = pos_det.iter().chain(neg_det.iter()).cloned().collect();
        let det_labels: Vec<f32> = [vec![1.0; 10], vec![0.0; 10]].concat();
        let det_sample_weights: Vec<f32> = [vec![10.0; 10], vec![1.0; 10]].concat();

        let (w1, b1) = train_logistic_sgd(
            &det_features,
            &det_labels,
            &det_sample_weights,
            L2_LAMBDA,
            LOGISTIC_LEARNING_RATE,
            LOGISTIC_MAX_ITER,
            Some(42),
        );
        let (w2, b2) = train_logistic_sgd(
            &det_features,
            &det_labels,
            &det_sample_weights,
            L2_LAMBDA,
            LOGISTIC_LEARNING_RATE,
            LOGISTIC_MAX_ITER,
            Some(42),
        );

        assert_eq!(
            w1, w2,
            "weights differ between deterministic logistic training runs"
        );
        assert!(
            (b1 - b2).abs() < f32::EPSILON,
            "bias differs between deterministic logistic training runs"
        );
    }

    fn make_positive_frame(rng: &mut impl Rng) -> Vec<f32> {
        (0..EMBEDDING_DIM)
            .map(|_| {
                loop {
                    let u1: f32 = rng.random();
                    let u2: f32 = rng.random();
                    if u1 > 0.0 && u2 > 0.0 {
                        let r = (-2.0 * u1.ln()).sqrt();
                        let theta = 2.0 * std::f32::consts::PI * u2;
                        break 0.5 + 0.3 * r * theta.cos();
                    }
                }
            })
            .collect()
    }

    /// Generate a synthetic 96-dim per-frame embedding with values clustered
    /// around -0.5 (simulates non-wake-word frame).
    fn make_negative_frame(rng: &mut impl Rng) -> Vec<f32> {
        (0..EMBEDDING_DIM)
            .map(|_| {
                loop {
                    let u1: f32 = rng.random();
                    let u2: f32 = rng.random();
                    if u1 > 0.0 && u2 > 0.0 {
                        let r = (-2.0 * u1.ln()).sqrt();
                        let theta = 2.0 * std::f32::consts::PI * u2;
                        break -0.5 + 0.3 * r * theta.cos();
                    }
                }
            })
            .collect()
    }

    // ── EmbeddingSequence cross-boundary tests (mahbot-902) ────────────────
    // These verify that training operates on per-sequence windows only, never
    // combining frames from different sequences.

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
        // → train_logistic_inner gets 0 positive windows + 0 negative windows → untrained.
        let verifier = VoiceVerifier::train(
            &[pos_seq],
            &[neg_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            L2_LAMBDA,
            Some(42),
        );
        assert!(
            !verifier.is_trained(),
            "Cross-sequence boundary window eliminated — each sequence < WINDOW_SIZE"
        );
    }

    #[test]
    fn test_verifier_train_sequences() {
        // Two positive sequences + two negative sequences each with enough
        // frames to form windows → trained verifier accepts positives and
        // rejects negatives.
        let mut rng = StdRng::seed_from_u64(42);

        // Positive sequence 1: 5 frames → 3 stride-1 windows
        let pos1: Vec<Vec<f32>> = (0..5).map(|_| make_positive_embedding(&mut rng)).collect();
        // Positive sequence 2: 5 frames → 3 stride-1 windows
        let pos2: Vec<Vec<f32>> = (0..5).map(|_| make_positive_embedding(&mut rng)).collect();
        // Negative sequence 1: 5 frames → 3 stride-1 windows
        let neg1: Vec<Vec<f32>> = (0..5).map(|_| make_negative_embedding(&mut rng)).collect();
        // Negative sequence 2: 5 frames → 3 stride-1 windows
        let neg2: Vec<Vec<f32>> = (0..5).map(|_| make_negative_embedding(&mut rng)).collect();

        let pos_seqs = [
            make_seq(
                pos1,
                crate::audio::embedding_sequence::LabelStratum::Positive,
            ),
            make_seq(
                pos2,
                crate::audio::embedding_sequence::LabelStratum::Positive,
            ),
        ];
        let neg_seqs = [
            make_seq(
                neg1,
                crate::audio::embedding_sequence::LabelStratum::Negative,
            ),
            make_seq(
                neg2,
                crate::audio::embedding_sequence::LabelStratum::Negative,
            ),
        ];

        let verifier = VoiceVerifier::train(
            &pos_seqs,
            &neg_seqs,
            None, // no per-negative weights
            DEFAULT_VERIFIER_THRESHOLD,
            L2_LAMBDA,
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
        let cache_pos: Vec<Vec<f32>> = (0..20)
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
            L2_LAMBDA,
            Some(42),
        );
        assert!(
            cache_verifier.is_trained(),
            "Cache-weighted verifier must be trained",
        );

        // Verify discrimination with per-negative-weights (mahbot-993).
        // For this distribution (20 pos windows, 30 neg windows with weights
        // [15.0, 10.0, 1.0] applied at line 482), the dynamic formula gives
        // class_weight ≈ (10×15 + 10×10 + 10×1)/20 = 13.0 (vs raw ratio 1.5).
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
}
