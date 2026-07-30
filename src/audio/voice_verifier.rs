//! Verifier for wake word false-trigger suppression.
//!
//! Implements a lightweight second-stage classifier that runs AFTER the
//! Conv1D classifier fires, as an additional AND gate.
//!
//! Uses a **logistic regression** (mahbot-901): 97-parameter L2-regularized
//! logistic regression on temporally mean-pooled 96-dim embeddings.
//! Mean-pools the 3-frame window to 96-dim before L2-norm, scaler, and
//! linear+sigmoid (~335× fewer parameters than the previous 3-layer MLP).
//!
//! When not trained, the verifier acts as a no-op (all frames pass).
//!
//! # Architecture
//!
//! Training pipeline: per-frame embeddings → windowing → L2-norm →
//! StandardScaler → train (logistic SGD with L2).  Inference is ~3μs per frame.
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
/// Reduced from 0.01 to 0.001 (mahbot-949) because the previous 0.01
/// regularization pulled weights to zero, overpowering the weak discriminative
/// signal in the homogeneous 96-dim embedding space.  The class-weighted loss
/// already provides regularization, and the 97-parameter model on ~200 training
/// examples does not overfit meaningfully.  The early stopping mechanism
/// (patience=100) provides additional protection against overfitting.
pub(crate) const L2_LAMBDA: f32 = 0.001;

/// Learning rate for logistic regression SGD training (mahbot-901).
///
/// Higher than MLP's LEARNING_RATE (0.001 tuned for Adam) because logistic
/// regression with plain SGD on a convex surface benefits from larger step
/// sizes.  Tested at lr=0.01 against the HARD-tier benchmark.
///
/// ## Interaction with class_weight (mahbot-993)
///
/// The dynamic class_weight formula (`Σ(neg_window_weights) / n_pos_windows`)
/// increases positive-sample gradient magnitude by ~13× vs the old raw-count
/// ratio when per-tier weights are active.  This approximately doubles the
/// total gradient per batch (~1.86×).  If the benchmark shows training
/// oscillation, reduce LR proportionally to ~0.005.  Conversely, the L2
/// regularisation (L2_LAMBDA=0.001) becomes relatively ~1.86× weaker vs the
/// data term — if scores saturate near 0 or 1, increase L2 slightly.
pub(crate) const LOGISTIC_LEARNING_RATE: f32 = 0.01;

/// Maximum iterations for logistic regression training.
///
/// Logistic converges faster than the MLP (convex optimization vs non-convex),
/// so 1000 iterations suffice.  The MLP needs 2000 iterations due to the deeper
/// non-linear layers.
pub(crate) const LOGISTIC_MAX_ITER: usize = 1000;

/// Fraction of sequences held out for validation (80/20 train/val split).
///
/// Split is per-sequence (avoiding data leakage from overlapping windows) with
/// stratification preserving both pos:neg ratio AND negative tier proportions
/// (confusable/unrelated/ambient/owner).  Falls back to per-window random split
/// when per-sequence split produces insufficient validation data.
pub(crate) const VALIDATION_SPLIT: f32 = 0.2;

/// Early stopping patience: stop if validation loss hasn't improved for this
/// many iterations.  Set to 100 (10% of the 1000-iteration budget), matching
/// the Conv1D classifier's relative patience ratio (15% of 100 epochs).
pub(crate) const EARLY_STOP_PATIENCE: usize = 100;

/// Log training and validation loss every N iterations.
const LOG_LOSS_INTERVAL: usize = 50;

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

/// Verifier for wake word false-trigger suppression (second-stage AND gate).
///
/// Uses a **logistic regression** (mahbot-901): 97-parameter L2-regularized
/// logistic regression on temporally mean-pooled 96-dim embeddings.
/// Mean-pools the 3-frame window to 96-dim before L2-norm, scaler, and
/// linear+sigmoid (~335× fewer parameters than the previous 3-layer MLP).
///
/// When `trained` is `false`, the verifier is a no-op (all frames pass with
/// score 1.0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceVerifier {
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

impl VoiceVerifier {
    /// Create an untrained verifier (no-op: all frames pass).
    #[must_use]
    pub fn untrained() -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            scaler_mean: Vec::new(),
            scaler_std: Vec::new(),
            threshold: DEFAULT_VERIFIER_THRESHOLD,
            trained: false,
        }
    }

    /// Returns `true` if this verifier has been trained and is ready for
    /// inference.
    ///
    /// Validates that logistic regression weights are the correct dimension
    /// (96), the scaler is present and matches, and the bias is finite.
    #[must_use]
    pub fn is_trained(&self) -> bool {
        if !self.trained {
            return false;
        }
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
            && (self.scaler_mean.len() != EMBEDDING_DIM || self.scaler_std.len() != EMBEDDING_DIM)
        {
            return false;
        }
        // Bias must be finite.
        self.bias.is_finite()
    }

    /// Predict the probability that the given window is a genuine wake word.
    ///
    /// Accepts either 288-dim (mean-pools internally to 96-dim) or 96-dim
    /// (already pooled, e.g. from training diagnostics) input.
    ///
    /// Returns a score in `[0.0, 1.0]`. When untrained, always returns `1.0`
    /// (no-op — all frames pass).
    #[must_use]
    pub fn predict(&self, embedding: &[f32]) -> f32 {
        if !self.is_trained() {
            return 1.0;
        }

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

    /// Train a new verifier from positive and negative
    /// [`EmbeddingSequence`](crate::audio::embedding_sequence::EmbeddingSequence)
    /// inputs.  Trains a logistic regression classifier with L2 regularization.
    ///
    /// Windows are formed **within** each sequence independently (never across
    /// sequences), preventing the cross-utterance window contamination that
    /// existed when training operated on flat `&[Vec<f32>]` lists (mahbot-902).
    /// Windows are mean-pooled to 96-dim (mahbot-901) and L2-normalized before
    /// training (mahbot-870).
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
        Self::train_logistic_inner(
            positive_sequences,
            negative_sequences,
            per_negative_sequence_weights,
            threshold,
            l2_lambda,
            LOGISTIC_LEARNING_RATE,
            LOGISTIC_MAX_ITER,
            rng_seed,
        )
    }

    /// Internal training: logistic regression on mean-pooled 96-dim windows.
    ///
    /// This is the single training path used by both
    /// [`train`](Self::train) and
    /// [`train_with_synthetic_negatives`](Self::train_with_synthetic_negatives).
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
        // Per-sequence metadata for stratified train/val split (mahbot-949).
        struct SeqInfo {
            start: usize,
            count: usize,
            is_positive: bool,
            source: crate::audio::embedding_sequence::Source,
        }

        // Early exit if either side has zero frames to avoid training on empty data.
        // Both positive and negative examples are required (mahbot-902).
        let total_pos_frames: usize = positive_sequences.iter().map(|s| s.embeddings.len()).sum();
        let total_neg_frames: usize = negative_sequences.iter().map(|s| s.embeddings.len()).sum();
        if total_pos_frames == 0 || total_neg_frames == 0 {
            warn!(
                "Cannot train verifier: need both positive ({total_pos_frames}) and negative ({total_neg_frames}) frames",
            );
            return Self::untrained();
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
        // Supports two input modes:
        // 1. Per-frame 96-dim input: form stride-1 windows via windowing functions.
        // 2. Pre-windowed 288-dim input (e.g., test data): use directly.
        //
        // Track per-sequence metadata for stratified train/val split.
        let mut seq_infos: Vec<SeqInfo> = Vec::new();
        let mut windows: Vec<Vec<f32>> = Vec::new();
        let mut window_labels: Vec<f32> = Vec::new();
        let mut window_weights: Vec<f32> = Vec::new();

        // Positive sequences
        for seq in positive_sequences {
            let seq_windows = form_sequence_windows(&seq.embeddings);
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
            let seq_windows = form_sequence_windows(&seq.embeddings);
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
            return Self::untrained();
        }

        // Class weight from window counts (not embedding-frame counts), adjusted
        // for per-tier negative weights (mahbot-993).
        //
        // The old formula used raw `n_neg_windows / n_pos_windows`, which assumes
        // every negative window contributes equally to the gradient.  But negative
        // sequences have per-tier weights (Ambient=1×, Owner=3×, Unrelated=10×,
        // Confusable=15×) already applied to `window_weights` (line 482).  The
        // positive class weight must therefore account for the total effective
        // weight of negatives: `Σ(neg_window_weights) / n_pos_windows`.  This
        // ensures each class contributes equally to the overall gradient.
        //
        // When all per-negative weights are 1.0 (or `None` was passed), this
        // reduces to `n_neg_windows / n_pos_windows` — the same as the old
        // formula, making the fix backward-compatible.
        //
        // Note: the Conv1D classifier (`train_classifier` in wake_word_classifier.rs)
        // does NOT have this issue because it uses a symmetric class-weighting
        // scheme: `pw = total / (2*np)` and `nw = total / (2*nn)`, ensuring each
        // class contributes exactly `total/2` to the loss, and it has no per-
        // negative-sequence weights that could unbalance the gradient.
        let class_weight = {
            let n_pw_f = n_pos_windows as f32;
            if n_pw_f > 0.0 {
                // Sum the effective weights of negative windows, which already
                // have per-tier weights applied (line 482).
                let neg_sum: f32 = window_weights[n_pos_windows..].iter().sum();
                neg_sum / n_pw_f
            } else {
                1.0
            }
        };
        for w in &mut window_weights[0..n_pos_windows] {
            *w = class_weight;
        }

        // L2-normalize
        for w in &mut windows {
            let norm = w.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
            for v in w.iter_mut() {
                *v /= norm;
            }
        }

        // If pre-windowed 288-dim input was provided, mean-pool to 96-dim
        // (logistic path, mahbot-901).
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

        // ── Train/validation split ──
        // Per-sequence stratified split: preserve pos:neg ratio AND negative
        // tier proportions.  Falls back to per-window random split if
        // per-sequence split produces insufficient validation data.
        let mut rng: StdRng = if let Some(seed) = rng_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(rand::random())
        };

        // Group sequence indices by class and source tier.
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

        // Shuffle and assign to train/val, stratified by group.
        let mut tr_seq_idx: Vec<usize> = Vec::new();
        let mut val_seq_idx: Vec<usize> = Vec::new();

        pos_indices.shuffle(&mut rng);
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
            shuffled.shuffle(&mut rng);
            let n_train = ((shuffled.len() as f32) * (1.0 - VALIDATION_SPLIT)).ceil() as usize;
            for (j, &idx) in shuffled.iter().enumerate() {
                if j < n_train {
                    tr_seq_idx.push(idx);
                } else {
                    val_seq_idx.push(idx);
                }
            }
        }

        // Build train/val arrays from the split sequence indices.
        let mut tr_windows: Vec<Vec<f32>> = Vec::new();
        let mut tr_labels: Vec<f32> = Vec::new();
        let mut tr_weights: Vec<f32> = Vec::new();
        let mut val_windows: Vec<Vec<f32>> = Vec::new();
        let mut val_labels: Vec<f32> = Vec::new();
        let mut val_weights: Vec<f32> = Vec::new();

        for &idx in &tr_seq_idx {
            let info = &seq_infos[idx];
            for i in info.start..info.start + info.count {
                tr_windows.push(windows[i].clone());
                tr_labels.push(window_labels[i]);
                tr_weights.push(window_weights[i]);
            }
        }
        for &idx in &val_seq_idx {
            let info = &seq_infos[idx];
            for i in info.start..info.start + info.count {
                val_windows.push(windows[i].clone());
                val_labels.push(window_labels[i]);
                val_weights.push(window_weights[i]);
            }
        }

        // If per-sequence split produced insufficient validation data, fall
        // back to per-window random split (matching Conv1D classifier behavior).
        let (
            tr_windows,
            tr_labels,
            tr_weights,
            val_windows,
            val_labels,
            val_weights,
            used_sequence_split,
        ) = if val_windows.is_empty() || tr_windows.len() < 2 {
            // Per-window fallback: random shuffle + 80/20 split
            let total = windows.len();
            let n_val = ((total as f32) * VALIDATION_SPLIT).ceil() as usize;
            let n_val = n_val.max(1).min(total - 1);
            let mut idx: Vec<usize> = (0..total).collect();
            idx.shuffle(&mut rng);
            let mut tr_f: Vec<Vec<f32>> = Vec::with_capacity(total - n_val);
            let mut tr_l: Vec<f32> = Vec::with_capacity(total - n_val);
            let mut tr_w: Vec<f32> = Vec::with_capacity(total - n_val);
            let mut va_f: Vec<Vec<f32>> = Vec::with_capacity(n_val);
            let mut va_l: Vec<f32> = Vec::with_capacity(n_val);
            let mut va_w: Vec<f32> = Vec::with_capacity(n_val);
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
            (
                tr_windows,
                tr_labels,
                tr_weights,
                val_windows,
                val_labels,
                val_weights,
                true,
            )
        };

        // ── Fit StandardScaler on positive training windows only ──
        // (mahbot-949 Fix 3: avoids mixing class distributions and reducing
        // inter-class separation before the logistic regression sees the data).
        let pos_tr_indices: Vec<usize> = tr_labels
            .iter()
            .enumerate()
            .filter(|&(_, &l)| l > 0.5)
            .map(|(i, _)| i)
            .collect();
        let pos_tr_windows: Vec<Vec<f32>> = pos_tr_indices
            .iter()
            .map(|&i| tr_windows[i].clone())
            .collect();
        let (scaler_mean, scaler_std) = if pos_tr_windows.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            compute_standard_scaler(&pos_tr_windows)
        };

        // Apply the scaler (fitted on positive training only) to ALL windows.
        // Matches inference: only scale dimensions with positive std; leave
        // zero-variance dimensions unchanged (mahbot-949).
        let apply_scaler = |fw: &[Vec<f32>]| -> Vec<Vec<f32>> {
            let use_scaler = !scaler_mean.is_empty() && !scaler_std.is_empty();
            fw.iter()
                .map(|f| {
                    let mut scaled = f.clone();
                    if use_scaler {
                        for (j, v) in scaled.iter_mut().enumerate() {
                            if scaler_std[j] > 0.0 {
                                *v = (*v - scaler_mean[j]) / scaler_std[j];
                            }
                        }
                    }
                    scaled
                })
                .collect()
        };

        let tr_scaled = apply_scaler(&tr_windows);
        let val_scaled = if val_windows.is_empty() {
            Vec::new()
        } else {
            apply_scaler(&val_windows)
        };

        // ── Train with validation, early stopping, and LR schedule ──
        let (weights, bias) = train_logistic_sgd_with_val(
            &tr_scaled,
            &tr_labels,
            &tr_weights,
            &val_scaled,
            &val_labels,
            &val_weights,
            l2_lambda,
            learning_rate,
            max_iter,
            EARLY_STOP_PATIENCE,
            rng_seed,
        );
        let verifier = Self {
            trained: true,
            weights,
            bias,
            scaler_mean,
            scaler_std,
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

    /// Log post-training diagnostics and check for discrimination collapse.
    ///
    /// Computes per-class mean scores on train and validation sets, emits an
    /// `info!` log, and warns if the validation pos/neg score ratio is below
    /// 1.1 (indicating a near-constant predictor — mahbot-949 Fix 6).
    #[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
    fn log_training_diagnostics(
        verifier: &Self,
        tr_windows: &[Vec<f32>],
        tr_labels: &[f32],
        val_windows: &[Vec<f32>],
        val_labels: &[f32],
        used_sequence_split: bool,
        class_weight: f32,
        l2_lambda: f32,
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

        // Validation scores (if val data exists)
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
            "Verifier training ({split_method} split): \
             train={n_tr_pos}+{n_tr_neg} val={n_val_pos}+{n_val_neg} windows, \
             class_weight={class_weight:.2}, L2={l2_lambda} | \
             train pos: mean={pos_mean_tr:.4} neg: mean={neg_mean_tr:.4} | \
             val pos: mean={pos_mean_val:.4} neg: mean={neg_mean_val:.4}"
        );

        // Discrimination check (mahbot-949 Fix 6): warn if val pos/neg
        // ratio is too low (indicating collapsed predictor).
        if !pos_scores_val.is_empty() && !neg_scores_val.is_empty() {
            let ratio = pos_mean_val / neg_mean_val.max(1e-10);
            if ratio < 1.1 {
                warn!(
                    "Verifier discrimination low: mean_pos_val/mean_neg_val={ratio:.4} (< 1.1). \
                     The verifier may have collapsed to a near-constant predictor."
                );
            }
        }
    }

    /// Convenience: train a verifier using the given positive embeddings and
    /// automatically generated synthetic negative examples (distribution-
    /// matched via [`generate_synthetic_negatives_from_positives`] instead of
    /// pure N(0,1) Gaussian noise).
    ///
    /// Uses logistic training hyperparameters (`LOGISTIC_MAX_ITER` /
    /// `LOGISTIC_LEARNING_RATE`).
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
        Self::train_logistic_inner(
            positive_sequences,
            &[synth_seq],
            None, // no per-negative weights for synthetic negatives
            threshold,
            L2_LAMBDA,
            LOGISTIC_LEARNING_RATE,
            LOGISTIC_MAX_ITER,
            rng_seed,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Window helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Form windows from a per-frame embedding list.
///
/// Always uses mean-pooled 96-dim windows (logistic path, mahbot-901).
///
/// Input can be either per-frame 96-dim embeddings (which get windowed)
/// or pre-windowed 288-dim data (which is mean-pooled to 96-dim and
/// L2-normalized).
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
/// (mahbot-901).  Instead of concatenating 3×96→288, it mean-pools to 96-dim,
/// preserving the same temporal context but reducing dimensionality for the
/// simpler logistic model.
///
/// Returns empty vec if fewer than 3 embeddings are available.
#[allow(clippy::cast_precision_loss)]
fn form_stride1_pooled_windows(embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
    stride1_windows_impl(embeddings, EMBEDDING_DIM, |i, out| {
        mean_pool_triple_into(&embeddings[i], &embeddings[i + 1], &embeddings[i + 2], out);
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

/// Compute per-dimension mean and population standard deviation for
/// StandardScaler normalisation (matching sklearn's `StandardScaler` with
/// default `ddof=0`).
///
/// Returns an empty `(Vec, Vec)` pair when `features` is empty.
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
        *s = (*s / n).sqrt();
        // Leave zero-variance dimensions at 0.0 — scaler will pass through
    }

    (mean, std)
}

// ═══════════════════════════════════════════════════════════════════════════
// Logistic regression SGD with validation + early stopping
// ═══════════════════════════════════════════════════════════════════════════

/// Train a logistic regression classifier with validation-based early stopping
/// and LR step decay (mahbot-949).
///
/// Same architecture as [`train_logistic_sgd`] but accepts separate validation
/// data and implements:
/// - Early stopping: stops if validation loss hasn't improved for `patience` steps
/// - Loss logging: logs train and val loss every `LOG_LOSS_INTERVAL` iterations
/// - LR step decay: halves the learning rate unconditionally every 200 iterations
///
/// When `val_features` is empty, runs without early stopping (full `max_iter`
/// iterations), matching the original `train_logistic_sgd` behavior.
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

/// Compute the weighted binary cross-entropy loss.
///
/// When `use_l2` is true, includes the L2 regularization term `(λ/2)·||w||²`.
/// Validation loss should NOT include the L2 term, since the regularization
/// penalty shrinks as weights decay and would artificially lower the validation
/// loss, biasing early stopping decisions (mahbot-949).
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

// ═══════════════════════════════════════════════════════════════════════════
// Logistic regression SGD (original, no validation)
// ═══════════════════════════════════════════════════════════════════════════

/// Train a logistic regression classifier on scaled 96-dim features using SGD
/// with L2 regularization (mahbot-901).
/// The cross-entropy loss with L2 penalty and sample weighting is:
/// ```text
/// J = -(1/N) Σ w_i · [y_i·log(σ_i) + (1-y_i)·log(1-σ_i)] + (λ/2)·||w||²
/// ```
///
/// Where `w_i` is the per-sample weight (includes class imbalance compensation),
/// and `||w||²` is the L2 norm of the weight vector (bias is not regularized).
///
/// Uses plain SGD (no momentum/Adam) since the convex logistic regression
/// landscape with 97 parameters doesn't need adaptive optimizers.
///
/// # Returns
/// `(weights, bias)` — the trained logistic regression parameters.
#[allow(
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
#[cfg_attr(not(test), expect(dead_code))]
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
#[must_use]
#[cfg_attr(not(test), expect(dead_code))]
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

        // Structural assertions: weights dimension, scaler, bias finite.
        assert_eq!(
            verifier.weights.len(),
            EMBEDDING_DIM,
            "verifier weights must be {EMBEDDING_DIM}-dim",
        );
        assert!(verifier.bias.is_finite(), "bias must be finite");
        assert_eq!(
            verifier.scaler_mean.len(),
            EMBEDDING_DIM,
            "scaler_mean must be {EMBEDDING_DIM}-dim",
        );
        assert_eq!(
            verifier.scaler_std.len(),
            EMBEDDING_DIM,
            "scaler_std must be {EMBEDDING_DIM}-dim",
        );

        // Verify discrimination on held-out per-frame input.
        let held_out_pos_frame = make_positive_frame(&mut rng);
        let held_out_neg_frame = make_negative_frame(&mut rng);
        let score_pos = verifier.predict(&held_out_pos_frame);
        let score_neg = verifier.predict(&held_out_neg_frame);
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
            L2_LAMBDA,                  // L2 regularization (mahbot-949: reduced to 0.001)
            Some(42),                   // deterministic seed for reproducibility
        );

        assert!(verifier.is_trained(), "Verifier must be trained");

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

        let verifier = VoiceVerifier::train_with_synthetic_negatives(
            &[pos_seq],
            DEFAULT_VERIFIER_THRESHOLD,
            Some(99),
        );

        assert!(verifier.is_trained(), "Verifier must be trained");
        assert_eq!(
            verifier.threshold, DEFAULT_VERIFIER_THRESHOLD,
            "threshold must match DEFAULT_VERIFIER_THRESHOLD",
        );

        // Structural assertions: weights dimension, scaler populated, finite.
        assert_eq!(
            verifier.weights.len(),
            EMBEDDING_DIM,
            "weights should be {EMBEDDING_DIM}-dim",
        );
        assert!(!verifier.scaler_mean.is_empty());
        assert!(!verifier.scaler_std.is_empty());
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
        // The key insight: even though the verifier was trained on
        // synthetic N(0,1) negatives (not real non-wake-word speech), the
        // N(0.5, 0.3) positives are sufficiently separated from N(0, 0.6)
        // speech to maintain a useful decision boundary at 0.5 for this
        // test. In production, the fallback is only triggered when <2 real
        // chunks are available, which is rare during normal enrollment.
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

        // Compute StandardScaler on POSITIVE L2-normalized features only
        // (mahbot-949 Fix 3: matching the updated training pipeline).
        let (scaler_mean, scaler_std) = compute_standard_scaler(&pos_norm);

        // Scale both positive and negative features.
        let scale_features = |fw: &[Vec<f32>]| -> Vec<Vec<f32>> {
            fw.iter()
                .map(|f| {
                    f.iter()
                        .enumerate()
                        .map(|(j, &val)| {
                            if !scaler_std.is_empty() && scaler_std[j] > 0.0 {
                                (val - scaler_mean[j]) / scaler_std[j]
                            } else {
                                val
                            }
                        })
                        .collect()
                })
                .collect()
        };
        let pos_scaled = scale_features(&pos_norm);
        let neg_scaled = scale_features(&neg_norm);

        let mut scaled = pos_scaled;
        scaled.extend(neg_scaled);

        let labels: Vec<f32> = [vec![1.0; 30], vec![0.0; 50]].concat();
        let sample_weights: Vec<f32> = [vec![3.0; 30], vec![1.0; 50]].concat();

        let (weights, bias) = train_logistic_sgd(
            &scaled,
            &labels,
            &sample_weights,
            L2_LAMBDA,
            0.01,
            500,
            Some(42),
        );

        // Build a logistic verifier as train() would.
        let verifier = VoiceVerifier {
            weights,
            bias,
            scaler_mean,
            scaler_std,
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
    }

    #[test]
    fn test_verifier_rejects_mismatched_scaler_dims() {
        // A verifier with trained=true but scaler dimensions that don't match
        // weights must be detected as untrained.
        let verifier = VoiceVerifier {
            trained: true,
            weights: vec![0.5; EMBEDDING_DIM],
            bias: 0.0,
            scaler_mean: vec![0.1; 48], // wrong dimension (48 ≠ 96)
            scaler_std: vec![0.2; 48],
            threshold: DEFAULT_VERIFIER_THRESHOLD,
        };
        assert!(
            !verifier.is_trained(),
            "Mismatched scaler dims should report untrained"
        );

        // Also test partial mismatch: only scaler_std populated.
        let verifier2 = VoiceVerifier {
            trained: true,
            weights: vec![0.5; EMBEDDING_DIM],
            bias: 0.0,
            scaler_mean: Vec::new(),
            scaler_std: vec![0.2; 48], // non-empty but mismatched
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
        // must produce identical logistic regression weights.
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
        assert_eq!(
            v1.weights, v2.weights,
            "weights differ between deterministic training runs"
        );
        assert!(
            (v1.bias - v2.bias).abs() < f32::EPSILON,
            "bias differs between deterministic training runs"
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
