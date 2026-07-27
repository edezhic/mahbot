//! Verifier for wake word false-trigger suppression.
//!
//! Implements a lightweight second-stage classifier that runs AFTER the
//! Conv1D MLP classifier fires, as an additional AND gate. Two architectures
//! are available, switchable via `MAHBOT_USE_LOGISTIC_VERIFIER` env var:
//!
//! - **MLP (default)**: 3-layer MLP (288 → 96 Leaky ReLU → 48 Leaky ReLU → 1
//!   sigmoid) with StandardScaler normalization (mahbot-861).  Leaky ReLU
//!   (slope 0.01) was adopted in mahbot-882 to prevent dead neuron issues.
//!   Operates on 3-frame stride-1 windows (288-dim = 3×96).
//! - **Logistic regression (mahbot-901)**: 97-parameter L2-regularized logistic
//!   regression on temporally mean-pooled 96-dim embeddings.  Mean-pools the
//!   3-frame window to 96-dim before L2-norm, scaler, and linear+sigmoid.
//!   ~335× fewer parameters than the MLP, less prone to overfitting.
//!
//! When not trained, the verifier acts as a no-op (all frames pass).
//!
//! # Architecture
//!
//! Both architectures share the same training pipeline (per-frame embeddings →
//! windowing → L2-norm → StandardScaler → train), differing only in window
//! formation (concatenated 288-dim vs mean-pooled 96-dim) and the classifier
//! itself (3-layer MLP with Adam vs linear SGD with L2).  Inference is
//! ~3μs per frame for either path.
//!
//! Backward compatibility: old logistic regression models (pre-mahbot-861) and
//! current MLP models are still supported for inference via `verifier_version`
//! discrimination — version 0 auto-detects via heuristic, version 1 uses the
//! logistic path explicitly.
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
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use tracing::{info, warn};

use crate::embedding_sequence::EmbeddingSequence;
use crate::wake_word_classifier::WakeWordClassifier;
use crate::{EMBEDDING_DIM, VERIFIER_INPUT_DIM, VERIFIER_WINDOW_SIZE};

/// Verifier architecture version: legacy auto-detect (MLP or linear).
pub(crate) const VERIFIER_VERSION_LEGACY: u8 = 0;

/// Verifier architecture version: logistic regression on mean-pooled 96-dim embeddings (mahbot-901).
pub(crate) const VERIFIER_VERSION_LOGISTIC: u8 = 1;

/// Define a cached env-var flag check function.
///
/// Generates a `$vis fn $name() -> bool` that checks `$env_var` once (then
/// caches the result in a `OnceLock<bool>`).  Values `"1"` or `"true"`
/// (case-insensitive) return `true`; all other values (including unset) return
/// `false`.
///
/// # Example
///
/// ```ignore
/// define_env_flag!(use_foo, "MAHBOT_USE_FOO");
/// define_env_flag!(pub(crate) use_bar, "MAHBOT_USE_BAR");
/// assert!(!use_foo());
/// ```
macro_rules! define_env_flag {
    ($vis:vis $name:ident, $env_var:expr) => {
        $vis fn $name() -> bool {
            static CACHED: OnceLock<bool> = OnceLock::new();
            *CACHED.get_or_init(|| match std::env::var($env_var) {
                Ok(val) => {
                    let v = val.trim().to_lowercase();
                    v == "1" || v == "true"
                }
                Err(_) => false,
            })
        }
    };
}

define_env_flag!(use_logistic_verifier, "MAHBOT_USE_LOGISTIC_VERIFIER");

/// Default decision threshold for the verifier (MLP decision boundary).
///
/// Calibrated empirically via threshold sweep (mahbot-890).  The sweep ran
/// HARD-tier benchmarks at 0.05 increments from 0.40 to 0.70 on the
/// **production cache path** (default — no `MAHBOT_BENCH_LEGACY_NEGATIVES`),
/// with top candidates (0.55, 0.60) replicated 4-5 additional runs to
/// estimate variance from stochastic training.  The selected threshold
/// maximises the mean detection rate while best satisfying all HARD-tier
/// false-accept constraints (confusable ≤1, noise ≤1, total ≤2).
///
/// ## Sweep results (mahbot-890, CONFUSABLE_UPWEIGHT=15, Leaky ReLU, production cache path)
///
/// | Threshold | Runs | Detection rate (range) | Mean DR | Verifier-pass FA / run | HARD (conf≤1, total≤2) pass rate |
/// |-----------|------|----------------------|---------|----------------------|----------------------------------|
/// | 0.40      | 1    | 92.3%                | 92.3%   | 4                     | ✗ (conf=2, total=4) |
/// | 0.45      | 1    | 84.6%                | 84.6%   | 3                     | ✗ (conf=1, total=3) |
/// | 0.50      | 1    | 53.8%                | 53.8%   | 2                     | ✗ (conf=1, total=2)† |
/// | 0.55      | 5    | 84.6–92.3%           | 89.2%   | 1.75                  | 3/5 (60%) |
/// | **0.60**  | 5    | **76.9–92.3%**       | **87.7%** | **1.0**              | **4/5 (80%)** |
/// | 0.65      | 3    | 84.6%                | 84.6%   | 1.0                   | 2/3 (67%) |
/// | 0.70      | 1    | 84.6%                | 84.6%   | 2                     | ✗ (conf=2, total=3) |
///
/// † The 0.50 run had 1 unrelated false accept (warm-up, verifier_score=0.000)
///   which violated unrel≤0, but conf≤1 and total≤2 were satisfied.
///   Warm-up false accepts (verifier inactive during first 4 embeddings) are a
///   classifier-side issue and appear across ALL thresholds; they are excluded
///   from the verifier-pass FA column.
///
/// **Selected: 0.60.**  Highest HARD-tier pass rate (80%) with best verifier-pass
/// FA control (1.0/run) and competitive mean detection rate (87.7%).  The small
/// DR trade-off (89.2%→87.7% vs 0.55) is justified by meaningfully lower
/// verifier-pass FA rate (1.0 vs 1.75/run).  The calibration against the
/// production cache path (pre-computed negative embeddings) verifies the
/// threshold for actual deployment — previous sweeps were run against the
/// legacy TTS negative synthesis path which produces a different verifier score
/// distribution.  MEDIUM and EASY tiers confirmed non-regressed vs 0.50.
///
/// ⚠ **If changing this constant**, re-run the HARD-tier calibration sweep
/// first: `MAHBOT_VERIFIER_THRESHOLD=<val> cargo bench --bench voice_pipeline_e2e`.
/// Then verify MEDIUM and EASY tiers at the new value.
///
/// Previously at 0.50 (mahbot-882), 0.4 (mahbot-853), 0.6 (mahbot-829),
/// 0.5 (mahbot-797), and 0.3 (mahbot-788).
pub(crate) const DEFAULT_VERIFIER_THRESHOLD: f32 = 0.60;

/// L2 regularization strength (lambda).
///
/// Reduced from 1.0 to 0.01 (mahbot-854) because the previous strong
/// regularization combined with extreme class imbalance (17:1 negatives-to-
/// positives) caused the model to learn constant near-zero outputs.  With
/// class-weighted loss now compensating for imbalance, weaker regularization
/// allows the model to develop discriminative weights.
pub(crate) const L2_LAMBDA: f32 = 0.01;

/// Learning rate for logistic regression SGD training (mahbot-901).
///
/// Higher than MLP's LEARNING_RATE (0.001 tuned for Adam) because logistic
/// regression with plain SGD on a convex surface benefits from larger step
/// sizes.  Tested at lr=0.01 against the HARD-tier benchmark.
pub(crate) const LOGISTIC_LEARNING_RATE: f32 = 0.01;

/// Maximum iterations for logistic regression training.
///
/// Logistic converges faster than the MLP (convex optimization vs non-convex),
/// so 1000 iterations suffice.  The MLP needs 2000 iterations due to the deeper
/// non-linear layers.
pub(crate) const LOGISTIC_MAX_ITER: usize = 1000;

/// Learning rate for gradient descent (Adam).
pub(crate) const LEARNING_RATE: f32 = 0.001;

/// Adam optimizer hyperparameters (mahbot-878).
/// Replaces plain SGD with Adam for more stable and faster convergence.
const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;

/// Maximum iterations for gradient descent.
///
/// Increased from 2,000 to 5,000 (mahbot-854) because class-weighted loss
/// requires more iterations to converge — the positive gradient signal is
/// amplified, making the loss landscape more complex.
///
/// Note: MLP training (mahbot-861) uses `MLP_MAX_ITER` (2000) instead since
/// the non-linear layers converge faster.  `MAX_ITER` is retained for the
/// `test_verifier_rejects_non_wake_speech` test which requires more iterations
/// for stable convergence with broad-cluster non-wake speech negatives.
#[cfg_attr(not(test), expect(dead_code))]
pub(crate) const MAX_ITER: usize = 5000;

/// MLP hidden layer 1 size (288 → 96).
pub(crate) const MLP_HIDDEN_1: usize = 96;

/// MLP hidden layer 2 size (96 → 48).
pub(crate) const MLP_HIDDEN_2: usize = 48;

/// Maximum iterations for MLP verifier training.
///
/// The MLP converges faster per-iteration than logistic regression because
/// the non-linear hidden layers provide richer gradient signal.  Set to 2000
/// as a balance between convergence quality and training latency (<1s with
/// ~2655 training examples and ~32,401 parameters).
pub(crate) const MLP_MAX_ITER: usize = 2000;

/// How much to upweight confusable negative examples during MLP training.
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

/// How much to upweight unrelated speech negative examples during MLP training.
///
/// Unrelated phrases (e.g. "what time is it", "good morning everyone") are
/// phonetically very different from the wake word but still represent real
/// non-wake-word speech that the verifier must reject.  10× gives them ~5×
/// more gradient contribution than ambient silence while still prioritising
/// confusable phrases as the primary negative signal.
pub(crate) const UNRELATED_UPWEIGHT: f32 = 10.0;

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
/// `cargo bench --bench voice_pipeline_e2e`.  The threshold sweep environmental
/// variable is `MAHBOT_VERIFIER_THRESHOLD`; there is currently no separate env
/// variable for this constant — adjust in source and re-benchmark.
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

/// Minimum number of mined negative windows required to use mining results
/// instead of falling back to all-windows training (mahbot-905).
///
/// If hard-negative mining produces fewer than this many total windows across
/// all source sequences, the miner falls back to the original all-windows
/// approach.  This prevents an untrained verifier when all negative utterances
/// produce near-zero classifier scores (e.g., silent or toy environments).
pub(crate) const MIN_MINED_NEGATIVES_FALLBACK: usize = 3;

define_env_flag!(pub(crate) use_hard_negative_mining, "MAHBOT_USE_HARD_NEGATIVE_MINING");

/// Result from hard-negative mining (mahbot-905).
///
/// Contains the mined [`EmbeddingSequence`] values (one per selected window)
/// and a summary count for diagnostics.
#[derive(Debug, Clone)]
pub(crate) struct MinedNegatives {
    /// Mined [`EmbeddingSequence`] values, one per selected window.
    ///
    /// Each sequence contains exactly [`VERIFIER_WINDOW_SIZE`] per-frame 96-dim
    /// embeddings, so `train_inner` forms exactly one stride-1 window from it.
    pub sequences: Vec<EmbeddingSequence>,
    /// Number of source sequences that contributed at least one mined window.
    pub source_sequences_represented: usize,
}

/// Mine hard-negative windows from negative [`EmbeddingSequence`] values using
/// a trained [`WakeWordClassifier`] (mahbot-905).
///
/// For each negative sequence, slides the classifier over all stride-1
/// 3-frame windows, records scores, and selects at most `max_per_sequence`
/// non-overlapping highest-scoring windows.  Windows are guaranteed to
/// share no embedding frames (separation ≥ `min_separation`).
///
/// This replaces the "all windows from all negatives" approach that dilutes
/// verifier training signal with hundreds of easy, non-discriminative windows.
///
/// # Algorithm
///
/// 1. For each [`EmbeddingSequence`], form all stride-1 windows.
/// 2. Score each window using `classifier.forward()`.
/// 3. Sort windows by score descending.
/// 4. Greedily select non-overlapping windows (minimum separation enforced),
///    at most `max_per_sequence` per source sequence.
/// 5. Package each selected window as a new [`EmbeddingSequence`] with exactly
///    [`VERIFIER_WINDOW_SIZE`] per-frame 96-dim embeddings.
///
/// # Returns
///
/// [`MinedNegatives`] — the mined sequences and a count of source sequences
/// that contributed.  Returns an empty `MinedNegatives` when all source
/// sequences are shorter than [`VERIFIER_WINDOW_SIZE`] frames.
///
/// # Panics
///
/// Panics if `min_separation < VERIFIER_WINDOW_SIZE`, because windows closer
/// than `VERIFIER_WINDOW_SIZE` frames share embeddings and would produce
/// overlapping training examples.
///
/// # Weight semantics
///
/// Mined negatives should be used with **uniform weights** (1.0) because the
/// mining process itself selects the hardest examples — the original tiered
/// per-sequence weights (ambient=1×, unrelated=10×, confusable=15×) were
/// designed to compensate for dilution by easy negatives.  With dilution
/// eliminated by mining, uniform weighting is appropriate and avoids the
/// pathological gradient concentration that would occur if each mined window
/// from a confusable sequence inherited the full 15× weight.
pub(crate) fn mine_hard_negatives(
    classifier: &WakeWordClassifier,
    negative_sequences: &[EmbeddingSequence],
    max_per_sequence: usize,
    min_separation: usize,
) -> MinedNegatives {
    assert!(
        min_separation >= VERIFIER_WINDOW_SIZE,
        "mine_hard_negatives: min_separation ({min_separation}) must be >= \
         VERIFIER_WINDOW_SIZE ({VERIFIER_WINDOW_SIZE})",
    );

    let mut sequences: Vec<EmbeddingSequence> = Vec::new();
    let mut source_sequences_represented: usize = 0;

    for seq in negative_sequences {
        let embeddings = &seq.embeddings;
        let n_frames = embeddings.len();

        // Need at least VERIFIER_WINDOW_SIZE frames to form any window.
        if n_frames < VERIFIER_WINDOW_SIZE {
            continue;
        }

        let n_windows = n_frames - VERIFIER_WINDOW_SIZE + 1;

        // Score each stride-1 window with the classifier.
        // The classifier.forward() takes &[Vec<f32>] of exactly
        // VERIFIER_WINDOW_SIZE per-frame 96-dim embeddings.
        let mut scored_windows: Vec<(f32, usize)> = Vec::with_capacity(n_windows);
        for i in 0..n_windows {
            let window_frames = &embeddings[i..i + VERIFIER_WINDOW_SIZE];
            let score = classifier.forward(window_frames);
            scored_windows.push((score, i));
        }

        // Sort by score descending.  Use partial_cmp with an Ordering fallback
        // to handle NaN scores gracefully (NaN is treated as equal to anything).
        scored_windows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Greedily select non-overlapping windows.
        let mut selected_indices: Vec<usize> = Vec::with_capacity(max_per_sequence);
        for &(_score, idx) in &scored_windows {
            if selected_indices.len() >= max_per_sequence {
                break;
            }
            // Check minimum separation from already-selected windows.
            let too_close = selected_indices.iter().any(|&sel_idx| {
                let dist = idx.abs_diff(sel_idx);
                dist < min_separation
            });
            if !too_close {
                selected_indices.push(idx);
            }
        }

        if !selected_indices.is_empty() {
            source_sequences_represented += 1;
        }

        // Build an EmbeddingSequence for each selected window.
        // Each sequence gets exactly VERIFIER_WINDOW_SIZE per-frame 96-dim
        // embeddings so that train_inner forms exactly one stride-1 window.
        for &idx in &selected_indices {
            let window_embs: Vec<Vec<f32>> = embeddings[idx..idx + VERIFIER_WINDOW_SIZE].to_vec();
            sequences.push(EmbeddingSequence::negative(
                seq.id.clone(),
                seq.source,
                seq.augmentation_family,
                window_embs,
            ));
        }
    }

    MinedNegatives {
        sequences,
        source_sequences_represented,
    }
}

/// Verifier for wake word false-trigger suppression (second-stage AND gate).
///
/// Two architectures are available, switchable via `MAHBOT_USE_LOGISTIC_VERIFIER`:
///
/// - **MLP (default, mahbot-861)**: 288 → 96 Leaky ReLU → 48 Leaky ReLU → 1
///   sigmoid, ~32,401 params.  Operates on 3-frame stride-1 windows (288-dim).
///   Computes `MLP(L2_norm(scaler(x)))`.
/// - **Logistic regression (mahbot-901)**: 97-param L2-regularized logistic
///   regression on temporally mean-pooled 96-dim embeddings.  Mean-pools the
///   3-frame window to 96-dim before L2-norm, scaler, and linear+sigmoid.
///   ~335× fewer parameters than the MLP, less prone to overfitting.
///
/// Both paths share the same training pipeline (per-frame embeddings →
/// windowing → L2-norm → StandardScaler → train), differing only in window
/// formation (concatenated 288-dim vs mean-pooled 96-dim) and classifier
/// (3-layer MLP with Adam vs linear SGD with L2).
///
/// When `trained` is `false`, the verifier is a no-op (all frames pass with
/// score 1.0).
///
/// Backward compatibility: old serialized models with `weights`+`bias` (linear,
/// pre-mahbot-861) that match the 288-dim input dimension are still supported
/// via the version-0 legacy auto-detect heuristic.  Legacy 96-dim linear models
/// are correctly rejected by `is_trained()` since they cannot process 288-dim
/// windows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceVerifier {
    /// Legacy logistic regression weights (backward compat, mahbot-777).
    #[serde(default)]
    pub weights: Vec<f32>,
    /// Legacy logistic regression bias (backward compat).
    #[serde(default)]
    pub bias: f32,

    // ── MLP weights (mahbot-861) ─────────────────────────────────────────
    // Architecture: 288 → 96 (Leaky ReLU, slope 0.01) → 48 (Leaky ReLU, slope 0.01) → 1 (sigmoid)
    //
    // Storage convention (row-major):
    //   w1[i * MLP_HIDDEN_1 + j] = weight from input[i] to hidden1[j]
    //   w2[j * MLP_HIDDEN_2 + k] = weight from hidden1[j] to hidden2[k]
    //   w3[k]                     = weight from hidden2[k] to output
    /// Layer 1 weights: 288 × 96 (row-major).
    #[serde(default)]
    pub w1: Vec<f32>,
    /// Layer 1 biases: 96.
    #[serde(default)]
    pub b1: Vec<f32>,
    /// Layer 2 weights: 96 × 48 (row-major).
    #[serde(default)]
    pub w2: Vec<f32>,
    /// Layer 2 biases: 48.
    #[serde(default)]
    pub b2: Vec<f32>,
    /// Layer 3 weights: 48 × 1.
    #[serde(default)]
    pub w3: Vec<f32>,
    /// Layer 3 bias (scalar).
    #[serde(default)]
    pub b3: f32,

    /// StandardScaler mean (per-dimension). Empty when scaling is not used.
    #[serde(default)]
    pub scaler_mean: Vec<f32>,
    /// StandardScaler std (per-dimension). Empty when scaling is not used.
    #[serde(default)]
    pub scaler_std: Vec<f32>,
    /// Decision threshold. Frames with a score below this are suppressed.
    #[serde(default = "default_verifier_threshold")]
    pub threshold: f32,
    /// Whether this verifier has been trained with positive + negative data.
    /// When true, expects 288-dim windowed inputs.
    #[serde(default)]
    pub trained: bool,
    /// Verifier architecture version.
    /// 0 = legacy auto-detect (MLP or linear, backward compat),
    /// 1 = logistic regression on mean-pooled 96-dim embeddings (mahbot-901).
    #[serde(default)]
    pub verifier_version: u8,
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
            w1: Vec::new(),
            b1: Vec::new(),
            w2: Vec::new(),
            b2: Vec::new(),
            w3: Vec::new(),
            b3: 0.0,
            scaler_mean: Vec::new(),
            scaler_std: Vec::new(),
            threshold: DEFAULT_VERIFIER_THRESHOLD,
            trained: false,
            verifier_version: VERIFIER_VERSION_LEGACY,
        }
    }

    /// Returns `true` if this verifier has been trained and is ready for
    /// inference.
    ///
    /// Validates that the model has either MLP parameters (new format,
    /// mahbot-861) or legacy linear weights (backward compat), and that
    /// scaler dimensions match the input dimension (288).  MLP weight/bias
    /// tensor sizes are also validated to prevent corrupted deserialization
    /// from reaching index-out-of-bounds in `mlp_forward()`.
    ///
    /// When both MLP (corrupted) and linear (valid) are present, the linear
    /// fallback is accepted — `predict()` will use `weights`/`bias` instead
    /// of the corrupted MLP parameters.
    #[must_use]
    pub fn is_trained(&self) -> bool {
        if !self.trained {
            return false;
        }

        // ── Version 1: logistic regression on mean-pooled 96-dim (mahbot-901) ──
        if self.verifier_version == VERIFIER_VERSION_LOGISTIC {
            // Must have 96-dim weights.
            if self.weights.len() != EMBEDDING_DIM {
                return false;
            }
            // Must have scaler at 96-dim.
            if self.scaler_mean.len() != EMBEDDING_DIM || self.scaler_std.len() != EMBEDDING_DIM {
                return false;
            }
            // Bias must be finite.
            return self.bias.is_finite();
        }

        // ── Version 0: legacy auto-detect (MLP or linear, backward compat) ──
        let has_valid_mlp = self.has_valid_mlp_params();
        let has_linear = !self.weights.is_empty();
        if !has_valid_mlp && !has_linear {
            return false;
        }
        // If falling back to linear (no valid MLP), validate that weights
        // dimension matches the expected input dimension.  A wrong-length
        // weights vector would silently produce truncated dot products via
        // zip() in predict().
        if !has_valid_mlp && has_linear && self.weights.len() != VERIFIER_INPUT_DIM {
            return false;
        }
        // Reject linear + scaler combination (mahbot-870).  predict() ignores
        // the scaler for linear models (legacy linear models predate both L2
        // normalization and the StandardScaler), so accepting this combination
        // would silently produce wrong predictions — the scaler is fitted and
        // stored but never applied.  Only the MLP path uses the scaler.
        if !has_valid_mlp
            && has_linear
            && (!self.scaler_mean.is_empty() || !self.scaler_std.is_empty())
        {
            return false;
        }
        // If either scaler is non-empty, both must be present and match the
        // 288-dim input.
        let input_dim = VERIFIER_INPUT_DIM;
        if (!self.scaler_mean.is_empty() || !self.scaler_std.is_empty())
            && (self.scaler_mean.len() != input_dim || self.scaler_std.len() != input_dim)
        {
            return false;
        }
        true
    }

    /// Returns `true` if MLP weight/bias tensor dimensions match the
    /// 288→96→48→1 architecture and `b3` is finite (no NaN/Inf which would
    /// produce `sigmoid(NaN)=0.5` silently).
    ///
    /// Only `b3` (the scalar output bias) is checked for finiteness: it's a
    /// single float, so the check is free, and a NaN there would corrupt every
    /// prediction (sigmoid output pinned to 0.5) — the most dangerous single
    /// failure point from serialization corruption.  Weight tensors are not
    /// individually validated for NaN/Inf because element-wise checks on ~32401
    /// floats would dominate the `is_trained()` hot path; dimension mismatches
    /// (caught above) cover most real-world serialization format errors.
    ///
    /// Used by both `is_trained()` and `predict()` to safely route between
    /// MLP and legacy linear inference paths.
    ///
    /// ⚠ **Temporary — will be removed after mahbot-901 benchmark validates logistic as default.**
    #[must_use]
    fn has_valid_mlp_params(&self) -> bool {
        if self.w1.is_empty() || self.b1.is_empty() {
            return false;
        }
        // Check tensor dimensions match the architecture.
        if self.w1.len() != VERIFIER_INPUT_DIM * MLP_HIDDEN_1
            || self.b1.len() != MLP_HIDDEN_1
            || self.w2.len() != MLP_HIDDEN_1 * MLP_HIDDEN_2
            || self.b2.len() != MLP_HIDDEN_2
            || self.w3.len() != MLP_HIDDEN_2
        {
            return false;
        }
        // Check b3 is finite (the other tensors are validated by the training
        // path, but a corrupted serialized model could have b3=NaN, which
        // would produce sigmoid(NaN) = 0.5 silently).
        self.b3.is_finite()
    }

    /// Threshold for verifier decision-making.
    ///
    /// Returns the value of [`DEFAULT_VERIFIER_THRESHOLD`] by default.
    /// Benchmarks and tests should reference this method instead of hardcoding
    /// a literal threshold value, because it respects the
    /// `MAHBOT_VERIFIER_THRESHOLD` env-var override.
    ///
    /// **Production code** should use [`DEFAULT_VERIFIER_THRESHOLD`] directly;
    /// the env-var override in this method exists solely for benchmark
    /// calibration sweeps (mahbot-880) and should not be relied upon in
    /// production paths.
    #[must_use]
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub(crate) fn default_threshold() -> f32 {
        // Allow env-var override for threshold calibration sweeps (mahbot-880).
        // Parsed once and cached to avoid repeated env var lookups.
        // NOTE: This uses OnceLock caching (reads once per process), unlike
        // MAHBOT_BENCH_LEGACY_NEGATIVES which is read on-demand in the benchmark.
        // The caching is intentional — the threshold is set once at process start and
        // should not change mid-run.  Use separate process invocations for per-test
        // overrides (or set env before test init).
        static CACHED_THRESHOLD: OnceLock<Option<f32>> = OnceLock::new();
        if let Some(threshold) =
            *CACHED_THRESHOLD.get_or_init(|| match std::env::var("MAHBOT_VERIFIER_THRESHOLD") {
                Ok(val) => {
                    if let Ok(t) = val.parse::<f32>() {
                        Some(t)
                    } else {
                        warn!(
                            "MAHBOT_VERIFIER_THRESHOLD='{val}' is not a valid f32 — \
                             using DEFAULT_VERIFIER_THRESHOLD ({})",
                            DEFAULT_VERIFIER_THRESHOLD,
                        );
                        None
                    }
                }
                Err(_) => None,
            })
        {
            return threshold;
        }
        DEFAULT_VERIFIER_THRESHOLD
    }

    /// Predict the probability that the given window is a genuine wake word.
    ///
    /// Accepts either:
    /// - 288-dim concatenated 3-frame window (MLP or legacy linear path), or
    /// - 96-dim mean-pooled window (logistic path, mahbot-901).
    ///
    /// Returns a score in `[0.0, 1.0]`. When untrained, always returns `1.0`
    /// (no-op — all frames pass).
    ///
    /// For the MLP path, operates on 3-frame stride-1 windows (288-dim = 3×96),
    /// matching the classifier's windowing convention (mahbot-870).  The logistic
    /// path mean-pools to 96-dim before inference.
    ///
    /// Routing: logistic version (verifier_version=1) → [`predict_logistic`],
    /// MLP (has_valid_mlp_params) → [`mlp_forward`], legacy linear
    /// (weights+bias, no scaler) → dot product + sigmoid.
    #[must_use]
    pub fn predict(&self, embedding: &[f32]) -> f32 {
        if !self.is_trained() {
            return 1.0;
        }

        // Logistic version: accepts either 288-dim (mean-pools internally) or
        // 96-dim (already pooled, e.g. from training diagnostics).
        if self.verifier_version == VERIFIER_VERSION_LOGISTIC {
            return predict_logistic(
                embedding,
                &self.weights,
                self.bias,
                &self.scaler_mean,
                &self.scaler_std,
            );
        }

        // Legacy/MLP path: must be 288-dim.
        if embedding.len() != VERIFIER_INPUT_DIM {
            warn!(
                "Verifier embedding dimension mismatch: got {}, expected {}; falling back to no-op",
                embedding.len(),
                VERIFIER_INPUT_DIM,
            );
            return 1.0;
        }

        // ═══════════════════════════════════════════════════════════════════
        // Legacy/MLP path: raw 288-dim window → L2-norm → scaler → MLP or linear
        // ═══════════════════════════════════════════════════════════════════
        //
        // Training convention (mahbot-870): form_stride1_windows()
        // L2-normalizes each 288-dim window first, then the scaler is
        // fitted on L2-normalized windows.  Inference must match the
        // same ordering so the MLP receives inputs from the same
        // distribution as training.
        //
        // Both intermediate buffers use stack-allocated arrays (288 f32s =
        // 1152 bytes each) instead of heap Vecs, since predict() is called
        // on every streaming inference frame (mahbot-874).

        // Step 1: L2-normalize the input window (unit-sphere projection).
        let norm_l2: f32 = embedding
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(1e-10);
        let mut x_l2 = [0.0f32; VERIFIER_INPUT_DIM];
        #[allow(clippy::cast_precision_loss)]
        for (i, &v) in embedding.iter().enumerate() {
            x_l2[i] = v / norm_l2;
        }

        // Step 2: Apply StandardScaler (per-dim mean/std centering) on
        // the L2-normalized values — same order as training pipeline.
        let x: [f32; VERIFIER_INPUT_DIM] =
            if !self.scaler_mean.is_empty() && !self.scaler_std.is_empty() {
                let mut scaled = [0.0f32; VERIFIER_INPUT_DIM];
                for i in 0..VERIFIER_INPUT_DIM {
                    let std = self.scaler_std[i];
                    scaled[i] = if std > 0.0 {
                        (x_l2[i] - self.scaler_mean[i]) / std
                    } else {
                        x_l2[i]
                    };
                }
                scaled
            } else {
                x_l2 // no scaler — use L2-normalized input directly
            };

        // Use MLP if available (new format, mahbot-861), fall back to legacy
        // linear model for backward compatibility with 288-dim linear models.
        if self.has_valid_mlp_params() {
            mlp_forward(
                &x, &self.w1, &self.b1, &self.w2, &self.b2, &self.w3, self.b3,
            )
        } else if !self.weights.is_empty() {
            // Legacy linear combination: z = w·x + b.
            // Note: legacy linear models predate L2-norm and scaler,
            // so use raw embedding directly (not the L2-normed/scaled x).
            let z: f32 = embedding
                .iter()
                .zip(self.weights.iter())
                .map(|(x, w)| x * w)
                .sum::<f32>()
                + self.bias;
            sigmoid(z)
        } else {
            unreachable!(
                "predict() called on trained verifier with neither MLP nor linear weights"
            );
        }
    }

    /// Train a new verifier from positive and negative
    /// [`EmbeddingSequence`](crate::embedding_sequence::EmbeddingSequence)
    /// inputs.  Trains a 3-layer MLP (default) or logistic regression
    /// (when `MAHBOT_USE_LOGISTIC_VERIFIER=1`) with L2 regularization.
    ///
    /// Windows are formed **within** each sequence independently (never across
    /// sequences), preventing the cross-utterance window contamination that
    /// existed when training operated on flat `&[Vec<f32>]` lists (mahbot-902).
    /// The MLP path concatenates 3-frame stride-1 windows to 288-dim; the
    /// logistic path mean-pools to 96-dim windows (mahbot-901).  Each window
    /// is L2-normalized before training (mahbot-870).
    ///
    /// MLP architecture: 288 → 96 (Leaky ReLU, slope 0.01) → 48 (Leaky ReLU, slope 0.01) → 1 (sigmoid), ~32,401 params.
    /// Logistic architecture: 97-param L2-regularized logistic regression on mean-pooled 96-dim embeddings.
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
    ///   `negative_sequences.len()`.  Positives are weighted by the automatic
    ///   `n_neg_windows / n_pos_windows` class weight, computed from window
    ///   counts rather than the old flat-list frame counts (mahbot-902).
    /// * `threshold` — Decision threshold (defaults to
    ///   [`DEFAULT_VERIFIER_THRESHOLD`] in production).
    /// * `l2_lambda` — L2 regularisation strength.
    /// * `learning_rate` — Gradient descent learning rate.
    /// * `max_iter` — Maximum gradient descent iterations.
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
        learning_rate: f32,
        max_iter: usize,
        rng_seed: Option<u64>,
    ) -> Self {
        let use_logistic = use_logistic_verifier();
        // Override hyperparameters for logistic (convex SGD with 97 params
        // converges well with higher LR and fewer iterations), matching the
        // dispatch in [`train_with_synthetic_negatives`].
        let (learning_rate, max_iter) = if use_logistic {
            (LOGISTIC_LEARNING_RATE, LOGISTIC_MAX_ITER)
        } else {
            (learning_rate, max_iter)
        };
        Self::train_inner(
            positive_sequences,
            negative_sequences,
            per_negative_sequence_weights,
            threshold,
            l2_lambda,
            learning_rate,
            max_iter,
            rng_seed,
            use_logistic,
        )
    }

    /// Internal training dispatch with explicit model selection.
    ///
    /// When `use_logistic` is true, uses mean-pooled windowing + logistic SGD.
    /// When false, uses concatenated windowing + 3-layer MLP with Adam.
    ///
    /// This is the single decision point for the verifier architecture — both
    /// [`train`](Self::train) (public, env-var-driven) and
    /// [`train_with_synthetic_negatives`](Self::train_with_synthetic_negatives)
    /// (hyperparameter-aware) delegate here with an explicit `use_logistic` flag,
    /// eliminating the coupling risk of independently checking the env var.
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::similar_names,
        clippy::cast_precision_loss
    )]
    fn train_inner(
        positive_sequences: &[EmbeddingSequence],
        negative_sequences: &[EmbeddingSequence],
        per_negative_sequence_weights: Option<&[f32]>,
        threshold: f32,
        l2_lambda: f32,
        learning_rate: f32,
        max_iter: usize,
        rng_seed: Option<u64>,
        use_logistic: bool,
    ) -> Self {
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

        let mut windows: Vec<Vec<f32>> = Vec::new();
        let mut window_labels: Vec<f32> = Vec::new();
        let mut window_weights: Vec<f32> = Vec::new();

        // Positive sequences
        for seq in positive_sequences {
            let seq_windows = form_sequence_windows(&seq.embeddings, use_logistic);
            for w in seq_windows {
                windows.push(w);
                window_labels.push(1.0);
                window_weights.push(0.0); // placeholder — set to class_weight below
            }
        }
        let n_pos_windows = windows.len();

        // Negative sequences
        for (i, seq) in negative_sequences.iter().enumerate() {
            let seq_windows = form_sequence_windows(&seq.embeddings, use_logistic);
            let seq_weight = weights_to_use.map_or(1.0, |pw| pw[i]);
            for w in seq_windows {
                windows.push(w);
                window_labels.push(0.0);
                window_weights.push(seq_weight);
            }
        }
        let n_neg_windows = window_labels.len() - n_pos_windows;

        if windows.is_empty() {
            warn!(
                "Cannot form windows: need at least {VERIFIER_WINDOW_SIZE} per-frame embeddings per sequence",
            );
            return Self::untrained();
        }

        // Class weight from window counts (not embedding-frame counts).
        //
        // The old flat-list approach windowed the combined positive+negative
        // embedding list and used n_neg_frames / n_pos_frames.  Here each
        // sequence is windowed independently, so sequences shorter than
        // VERIFIER_WINDOW_SIZE produce zero windows.  Window counts and
        // frame counts therefore diverge for short sequences.  Using window
        // counts is correct — each window is one training example whose class
        // weight represents the inverse prevalence of its label (mahbot-902).
        let class_weight = {
            let n_pw_f = n_pos_windows as f32;
            let n_nw_f = n_neg_windows as f32;
            if n_pw_f > 0.0 { n_nw_f / n_pw_f } else { 1.0 }
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

        let mut use_logistic = use_logistic;
        if use_logistic && windows[0].len() != EMBEDDING_DIM {
            warn!(
                "Logistic mode but {}-dim features; falling back to MLP",
                windows[0].len()
            );
            use_logistic = false;
        }

        let (scaler_mean, scaler_std) = compute_standard_scaler(&windows);
        let mut scaled = windows.clone();
        for w in &mut scaled {
            for (j, v) in w.iter_mut().enumerate() {
                let std = scaler_std[j].max(1e-10);
                *v = (*v - scaler_mean[j]) / std;
            }
        }
        let input_dim = scaled[0].len();

        let verifier = if use_logistic {
            let (weights, bias) = train_logistic_sgd(
                &scaled,
                &window_labels,
                &window_weights,
                l2_lambda,
                learning_rate,
                max_iter,
                rng_seed,
            );
            Self {
                trained: true,
                verifier_version: VERIFIER_VERSION_LOGISTIC,
                weights,
                bias,
                w1: Vec::new(),
                b1: Vec::new(),
                w2: Vec::new(),
                b2: Vec::new(),
                w3: Vec::new(),
                b3: 0.0,
                scaler_mean,
                scaler_std,
                threshold,
            }
        } else {
            let MlpWeights {
                w1,
                b1,
                w2,
                b2,
                w3,
                b3,
            } = train_mlp(
                &scaled,
                &window_labels,
                input_dim,
                &window_weights,
                l2_lambda,
                learning_rate,
                max_iter,
                rng_seed,
            );
            Self {
                trained: true,
                verifier_version: VERIFIER_VERSION_LEGACY,
                weights: Vec::new(),
                bias: 0.0,
                w1,
                b1,
                w2,
                b2,
                w3,
                b3,
                scaler_mean,
                scaler_std,
                threshold,
            }
        };

        // Diagnostics
        {
            let mut pos_scores = Vec::with_capacity(n_pos_windows);
            let mut neg_scores = Vec::with_capacity(n_neg_windows);
            for (emb, &lbl) in windows.iter().zip(window_labels.iter()) {
                let score = verifier.predict(emb);
                if lbl > 0.5 {
                    pos_scores.push(score);
                } else {
                    neg_scores.push(score);
                }
            }
            let pos_mean = if pos_scores.is_empty() {
                0.0
            } else {
                pos_scores.iter().sum::<f32>() / n_pos_windows as f32
            };
            let neg_mean = if neg_scores.is_empty() {
                0.0
            } else {
                neg_scores.iter().sum::<f32>() / n_neg_windows as f32
            };
            let pos_min = pos_scores.iter().copied().fold(f32::INFINITY, f32::min);
            let pos_max = pos_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let neg_min = neg_scores.iter().copied().fold(f32::INFINITY, f32::min);
            let neg_max = neg_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            info!(
                "Verifier training: {n_pos_windows} pos + {n_neg_windows} neg windows, class_weight={class_weight:.2}, L2={l2_lambda} | pos: mean={pos_mean:.4} [{pos_min:.4},{pos_max:.4}] neg: mean={neg_mean:.4} [{neg_min:.4},{neg_max:.4}]"
            );
        }

        verifier
    }

    /// Convenience: train a verifier using the given positive embeddings and
    /// automatically generated synthetic negative examples (distribution-
    /// matched via [`generate_synthetic_negatives_from_positives`] instead of
    /// pure N(0,1) Gaussian noise).
    ///
    /// Uses appropriate training hyperparameters based on the active verifier
    /// model: `LOGISTIC_MAX_ITER` / `LOGISTIC_LEARNING_RATE` when logistic is
    /// enabled, or `MLP_MAX_ITER` / `LEARNING_RATE` for the MLP path.
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
        let use_logistic = use_logistic_verifier();
        let (max_iter, learning_rate) = if use_logistic {
            (LOGISTIC_MAX_ITER, LOGISTIC_LEARNING_RATE)
        } else {
            (MLP_MAX_ITER, LEARNING_RATE)
        };
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
            crate::embedding_sequence::UtteranceId {
                sequence_index: 0,
                variant_index: 0,
            },
            crate::embedding_sequence::Source::Synthetic,
            None,
            negatives,
        );
        Self::train_inner(
            positive_sequences,
            &[synth_seq],
            None, // no per-negative weights for synthetic negatives
            threshold,
            L2_LAMBDA,
            learning_rate,
            max_iter,
            rng_seed,
            use_logistic,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Window helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Form windows from a per-frame embedding list, dispatching to either
/// stride-1 concatenated windows (MLP) or mean-pooled windows (logistic).
///
/// Input can be either per-frame 96-dim embeddings (which get windowed)
/// or pre-windowed 288-dim data (which is L2-normalized and used directly).
fn form_sequence_windows(embeddings: &[Vec<f32>], use_logistic: bool) -> Vec<Vec<f32>> {
    if embeddings.is_empty() {
        return Vec::new();
    }
    if embeddings[0].len() == EMBEDDING_DIM {
        // Per-frame: form stride-1 windows.
        if use_logistic {
            form_stride1_pooled_windows(embeddings)
        } else {
            form_stride1_windows(embeddings)
        }
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

/// Form stride-1 windows from a flat list of 96-dim embeddings.
///
/// Each window is 3 consecutive embeddings concatenated into a 288-dim vector,
/// then L2-normalized (matching classifier convention).  Consecutive windows
/// overlap by 2 embeddings (stride 1).
///
/// Returns empty vec if fewer than 3 embeddings are available.
fn form_stride1_windows(embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
    stride1_windows_impl(embeddings, VERIFIER_INPUT_DIM, |i, out| {
        fill_verifier_window(embeddings, i, out);
    })
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
// MLP inference
// ═══════════════════════════════════════════════════════════════════════════

/// Forward pass through the 3-layer MLP (288 → 96 Leaky ReLU → 48 Leaky ReLU → 1 sigmoid).
///
/// ⚠ **Temporary — will be removed after mahbot-901 benchmark validation.**
///
/// # Panics
///
/// May panic (index out of bounds) if tensor sizes don't match the
/// architecture — callers should validate dimensions via [`VoiceVerifier::is_trained`]
/// before calling this function.
fn mlp_forward(
    x: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    w3: &[f32],
    b3: f32,
) -> f32 {
    let h1_size = MLP_HIDDEN_1;
    let h2_size = MLP_HIDDEN_2;
    let input_dim = x.len();

    // Layer 1: h1 = LeakyReLU(W1^T · x + b1, slope=0.01)
    // w1 is stored row-major: w1[i * h1_size + j] = weight from x[i] to h1[j]
    // So h1[j] = sum_i x[i] * w1[i * h1_size + j] + b1[j]
    let mut h1 = vec![0.0; h1_size];
    for j in 0..h1_size {
        let mut s = b1[j];
        for i in 0..input_dim {
            s += x[i] * w1[i * h1_size + j];
        }
        h1[j] = if s > 0.0 { s } else { 0.01 * s }; // Leaky ReLU (mahbot-882)
    }

    // Layer 2: h2 = LeakyReLU(W2^T · h1 + b2, slope=0.01)
    let mut h2 = vec![0.0; h2_size];
    for k in 0..h2_size {
        let mut s = b2[k];
        for j in 0..h1_size {
            s += h1[j] * w2[j * h2_size + k];
        }
        h2[k] = if s > 0.0 { s } else { 0.01 * s }; // Leaky ReLU (mahbot-882)
    }

    // Layer 3: out = sigmoid(W3^T · h2 + b3)
    let mut z = b3;
    for k in 0..h2_size {
        z += h2[k] * w3[k];
    }

    sigmoid(z)
}

// ═══════════════════════════════════════════════════════════════════════════
// MLP training
// ═══════════════════════════════════════════════════════════════════════════

/// Trained MLP weights and biases for the 288→96→48→1 voice verifier
/// (mahbot-861).  Returned by [`train_mlp`] to provide compile-time
/// argument-order safety when assigning fields to the [`VoiceVerifier`].
///
/// ⚠ **Temporary — will be removed after mahbot-901 benchmark validates logistic as default.**
struct MlpWeights {
    /// 288 × 96, row-major: `w1[i * 96 + j]` = weight from input `i` to h1 `j`.
    w1: Vec<f32>,
    /// 96 bias terms for h1.
    b1: Vec<f32>,
    /// 96 × 48, row-major: `w2[j * 48 + k]` = weight from h1 `j` to h2 `k`.
    w2: Vec<f32>,
    /// 48 bias terms for h2.
    b2: Vec<f32>,
    /// 48 × 1, flat: `w3[k]` = weight from h2 `k` to output.
    w3: Vec<f32>,
    /// Scalar output bias.
    b3: f32,
}

/// Train a logistic regression classifier on scaled 96-dim features using SGD
/// with L2 regularization (mahbot-901).
///
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
#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
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

/// Train a small MLP (288 → 96 → 48 → 1) using batch gradient descent with
/// L2 regularization and per-sample weighting (mahbot-861).
///
/// ⚠ **Temporary — will be removed after mahbot-901 benchmark validates logistic as default.**
///
/// The cross-entropy loss with L2 penalty and sample weighting is:
/// ```text
/// J = -(1/N) Σ w_i · [y_i·log(σ_i) + (1-y_i)·log(1-σ_i)] + (λ/2)·||W||²_F
/// ```
///
/// where `w_i` is the per-sample weight (includes class imbalance compensation
/// and confusable-phrase upweighting), and `||W||²_F` is the Frobenius norm
/// of all weight matrices (biases are not regularized).
///
/// Gradient averaging and L2 penalty follow sklearn conventions:
/// * Gradients are averaged over the batch (division by N).
/// * L2 penalty applied to weight matrices only (not biases).
/// * Xavier uniform initialization for all weights.
///
/// # Returns
/// [`MlpWeights`] — the trained MLP parameters with named fields.
#[allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
fn train_mlp(
    features: &[Vec<f32>],  // scaled (n × dim)
    labels: &[f32],         // 0.0 or 1.0
    dim: usize,             // input dimension (288)
    sample_weights: &[f32], // per-sample weight (n)
    l2_lambda: f32,
    learning_rate: f32,
    max_iter: usize,
    rng_seed: Option<u64>, // deterministic training when Some
) -> MlpWeights {
    let n = features.len();
    let h1_size = MLP_HIDDEN_1; // 96
    let h2_size = MLP_HIDDEN_2; // 48

    if n == 0 || dim == 0 {
        return MlpWeights {
            w1: Vec::new(),
            b1: Vec::new(),
            w2: Vec::new(),
            b2: Vec::new(),
            w3: Vec::new(),
            b3: 0.0,
        };
    }

    // ── Initialize weights with Xavier uniform init ──
    // w1: 288 × 96, stored row-major: w1[i * h1_size + j]
    let mut w1 = vec![0.0; dim * h1_size];
    let mut b1 = vec![0.0; h1_size];
    // w2: 96 × 48, stored row-major: w2[j * h2_size + k]
    let mut w2 = vec![0.0; h1_size * h2_size];
    let mut b2 = vec![0.0; h2_size];
    // w3: 48 × 1
    let mut w3 = vec![0.0; h2_size];
    let mut b3 = 0.0;

    // Xavier/Glorot uniform bound: sqrt(6 / (fan_in + fan_out))
    let w1_bound = (6.0 / (dim as f32 + h1_size as f32)).sqrt();
    let w2_bound = (6.0 / (h1_size as f32 + h2_size as f32)).sqrt();
    let w3_bound = (6.0 / (h2_size as f32 + 1.0)).sqrt();

    // Create seeded or entropy-based RNG for weight initialization.
    let mut rng: StdRng = if let Some(seed) = rng_seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::seed_from_u64(rand::random())
    };

    for w in &mut w1 {
        *w = rng.random::<f32>() * 2.0 * w1_bound - w1_bound;
    }
    for w in &mut w2 {
        *w = rng.random::<f32>() * 2.0 * w2_bound - w2_bound;
    }
    for w in &mut w3 {
        *w = rng.random::<f32>() * 2.0 * w3_bound - w3_bound;
    }

    // Pre-allocate work vectors
    let mut h1_pre = vec![0.0; h1_size];
    let mut h1 = vec![0.0; h1_size];
    let mut h2_pre = vec![0.0; h2_size];
    let mut h2 = vec![0.0; h2_size];
    let mut dh2 = vec![0.0; h2_size];
    let mut dh1 = vec![0.0; h1_size];

    let n_f32 = n as f32;

    // Gradient buffers (declared once, zeroed each iteration to avoid
    // re-allocation churn: ~259 MB over 2000 iterations with ~32,401 params).
    let mut dw1 = vec![0.0; dim * h1_size];
    let mut db1 = vec![0.0; h1_size];
    let mut dw2 = vec![0.0; h1_size * h2_size];
    let mut db2 = vec![0.0; h2_size];
    let mut dw3 = vec![0.0; h2_size];

    // ── Adam optimizer state (mahbot-878) ──
    // Replaces plain SGD with Adam for stable convergence.
    // Each parameter group has its own momentum (m) and velocity (v).
    let mut adam_t: usize = 0;
    // w1: dim × h1_size
    let mut adam_mmt_w1 = vec![0.0; w1.len()];
    let mut adam_vel_w1 = vec![0.0; w1.len()];
    let mut adam_mmt_b1 = vec![0.0; b1.len()];
    let mut adam_vel_b1 = vec![0.0; b1.len()];
    // w2: h1_size × h2_size
    let mut adam_mmt_w2 = vec![0.0; w2.len()];
    let mut adam_vel_w2 = vec![0.0; w2.len()];
    let mut adam_mmt_b2 = vec![0.0; b2.len()];
    let mut adam_vel_b2 = vec![0.0; b2.len()];
    // w3: h2_size
    let mut adam_mmt_w3 = vec![0.0; w3.len()];
    let mut adam_vel_w3 = vec![0.0; w3.len()];
    let mut adam_mmt_b3 = 0.0;
    let mut adam_vel_b3 = 0.0;

    for _iteration in 0..max_iter {
        // Zero gradient buffers for this iteration.
        dw1.fill(0.0);
        db1.fill(0.0);
        dw2.fill(0.0);
        db2.fill(0.0);
        dw3.fill(0.0);
        let mut db3 = 0.0;

        for i in 0..n {
            let x = &features[i];
            let y = labels[i];
            let w = sample_weights[i];

            // ── Forward pass ──────────────────────────────────────────────

            // h1 = LeakyReLU(W1^T · x + b1, slope=0.01)
            for j in 0..h1_size {
                let mut s = b1[j];
                for k in 0..dim {
                    s += x[k] * w1[k * h1_size + j];
                }
                h1_pre[j] = s;
                h1[j] = if s > 0.0 { s } else { 0.01 * s }; // Leaky ReLU (mahbot-882)
            }

            // h2 = LeakyReLU(W2^T · h1 + b2, slope=0.01)
            for k in 0..h2_size {
                let mut s = b2[k];
                for j in 0..h1_size {
                    s += h1[j] * w2[j * h2_size + k];
                }
                h2_pre[k] = s;
                h2[k] = if s > 0.0 { s } else { 0.01 * s }; // Leaky ReLU (mahbot-882)
            }

            // out = sigmoid(W3^T · h2 + b3)
            let mut z = b3;
            for k in 0..h2_size {
                z += h2[k] * w3[k];
            }
            let pred = sigmoid(z);

            // ── Backward pass ─────────────────────────────────────────────

            // Gradient of binary cross-entropy w.r.t. output logit:
            // dL/dz = (pred - y); weighted: w * (pred - y)
            let dz = w * (pred - y);

            // Layer 3 (output): dL/dW3[k] = dL/dz * h2[k]
            for k in 0..h2_size {
                dw3[k] += dz * h2[k];
            }
            db3 += dz;

            // Backprop to h2: dh2[k] = dL/dz * W3[k]
            for k in 0..h2_size {
                dh2[k] = dz * w3[k];
                // Leaky ReLU derivative: d(h2)/d(pre) = 1 if pre > 0 else 0.01
                if h2_pre[k] <= 0.0 {
                    dh2[k] *= 0.01; // Leaky ReLU (mahbot-882)
                }
            }

            // Layer 2: dL/dW2[j * h2_size + k] = dh2[k] * h1[j]
            for j in 0..h1_size {
                for k in 0..h2_size {
                    dw2[j * h2_size + k] += dh2[k] * h1[j];
                }
            }
            for k in 0..h2_size {
                db2[k] += dh2[k];
            }

            // Backprop to h1: dh1[j] = sum_k dh2[k] * W2[j * h2_size + k]
            for j in 0..h1_size {
                let mut s = 0.0;
                for k in 0..h2_size {
                    s += dh2[k] * w2[j * h2_size + k];
                }
                dh1[j] = s;
                // Leaky ReLU derivative (mahbot-882)
                if h1_pre[j] <= 0.0 {
                    dh1[j] *= 0.01; // Leaky ReLU slope
                }
            }

            // Layer 1: dL/dW1[i * h1_size + j] = dh1[j] * x[i]
            for i_idx in 0..dim {
                for j in 0..h1_size {
                    dw1[i_idx * h1_size + j] += dh1[j] * x[i_idx];
                }
            }
            for j in 0..h1_size {
                db1[j] += dh1[j];
            }
        }

        // ── Average over batch and add L2 regularization ──
        // (biases are not regularized, matching sklearn convention)
        for w in &mut dw1 {
            *w /= n_f32;
        }
        for w in &mut dw2 {
            *w /= n_f32;
        }
        for w in &mut dw3 {
            *w /= n_f32;
        }
        for d in &mut db1 {
            *d /= n_f32;
        }
        for d in &mut db2 {
            *d /= n_f32;
        }
        db3 /= n_f32;

        // Add L2 regularization gradient: λ * w
        for j in 0..w1.len() {
            dw1[j] += l2_lambda * w1[j];
        }
        for j in 0..w2.len() {
            dw2[j] += l2_lambda * w2[j];
        }
        for j in 0..w3.len() {
            dw3[j] += l2_lambda * w3[j];
        }

        // ── Adam parameter update (mahbot-878) ──
        adam_t += 1;
        // Bias-corrected learning rate: lr * sqrt(1 - b2^t) / (1 - b1^t)
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let lr_t = learning_rate * (1.0 - ADAM_BETA2.powi(adam_t as i32)).sqrt()
            / (1.0 - ADAM_BETA1.powi(adam_t as i32));

        // Helper closure for Adam update on a parameter slice.
        let adam_update = |p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32], lr_t: f32| {
            for i in 0..p.len() {
                m[i] = ADAM_BETA1 * m[i] + (1.0 - ADAM_BETA1) * g[i];
                v[i] = ADAM_BETA2 * v[i] + (1.0 - ADAM_BETA2) * g[i] * g[i];
                p[i] -= lr_t * m[i] / (v[i].sqrt() + ADAM_EPS);
            }
        };

        adam_update(&mut w1, &dw1, &mut adam_mmt_w1, &mut adam_vel_w1, lr_t);
        adam_update(&mut b1, &db1, &mut adam_mmt_b1, &mut adam_vel_b1, lr_t);
        adam_update(&mut w2, &dw2, &mut adam_mmt_w2, &mut adam_vel_w2, lr_t);
        adam_update(&mut b2, &db2, &mut adam_mmt_b2, &mut adam_vel_b2, lr_t);
        adam_update(&mut w3, &dw3, &mut adam_mmt_w3, &mut adam_vel_w3, lr_t);
        // b3 is a scalar — use single-element slices
        let b3_grad = db3;
        adam_mmt_b3 = ADAM_BETA1 * adam_mmt_b3 + (1.0 - ADAM_BETA1) * b3_grad;
        adam_vel_b3 = ADAM_BETA2 * adam_vel_b3 + (1.0 - ADAM_BETA2) * b3_grad * b3_grad;
        b3 -= lr_t * adam_mmt_b3 / (adam_vel_b3.sqrt() + ADAM_EPS);
    }

    MlpWeights {
        w1,
        b1,
        w2,
        b2,
        w3,
        b3,
    }
}

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
/// now used by [`validate_enrollment_consistency`](crate::voice::validate_enrollment_consistency)
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
/// Used by production [`finalize_enrollment`](crate::voice::finalize_enrollment)
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

    /// Predict from a pre-scaled embedding, skipping the StandardScaler
    /// and L2 normalization steps.  The input must be the fully processed
    /// 288-dim window (L2-normalized then StandardScaler-applied, matching
    /// the training pipeline order, mahbot-870).  No additional
    /// preprocessing is performed.
    ///
    /// **Requires MLP parameters** (mahbot-870): `predict_scaled()` is a
    /// test-only helper that bypasses preprocessing.  It only supports the
    /// MLP inference path — legacy linear models (which predate both L2
    /// normalization and the StandardScaler) are not supported through this
    /// helper because they would need raw input, not processed input.
    /// Callers must ensure the verifier has valid MLP parameters before
    /// calling this function.
    ///
    /// ⚠ **Temporary — will be removed after mahbot-901 benchmark validates logistic as default.**
    fn predict_scaled(verifier: &VoiceVerifier, scaled: &[f32]) -> f32 {
        debug_assert!(
            verifier.has_valid_mlp_params(),
            "predict_scaled requires MLP parameters; linear models are not supported",
        );
        mlp_forward(
            scaled,
            &verifier.w1,
            &verifier.b1,
            &verifier.w2,
            &verifier.b2,
            &verifier.w3,
            verifier.b3,
        )
    }

    /// Helper: wrap flat embeddings into a single EmbeddingSequence for testing.
    fn make_seq(
        embs: Vec<Vec<f32>>,
        label: crate::embedding_sequence::LabelStratum,
    ) -> EmbeddingSequence {
        EmbeddingSequence {
            id: crate::embedding_sequence::UtteranceId {
                sequence_index: 0,
                variant_index: 0,
            },
            source: crate::embedding_sequence::Source::Enrollment,
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
    /// audio that survives Conv1D MLP matching).  Unlike the old opposite-direction
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
    }

    #[test]
    fn assert_weight_tier_empty_tier() {
        // Edge case: count=0 should not panic at any offset
        let weights: Vec<f32> = vec![1.0, 2.0, 3.0];
        assert_weight_tier(&weights, 0, 0, 1.0, "empty-at-start");
        assert_weight_tier(&weights, 1, 0, 0.0, "empty-at-middle");
        assert_weight_tier(&weights, 3, 0, 0.0, "empty-at-end");
    }

    #[test]
    #[should_panic(
        expected = "Weight tier mismatch: first weight at position 2 should be 1, got 2"
    )]
    fn assert_weight_tier_mismatch_panics() {
        // Mismatch: should panic with descriptive message
        let weights = vec![1.0, 1.0, 2.0];
        assert_weight_tier(&weights, 0, 3, 1.0, "first");
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
        let pos_seq = make_seq(positives, crate::embedding_sequence::LabelStratum::Positive);
        let neg_seq = make_seq(negatives, crate::embedding_sequence::LabelStratum::Negative);

        let verifier = VoiceVerifier::train(
            &[pos_seq],
            &[neg_seq],
            None,                       // no per-negative weights
            DEFAULT_VERIFIER_THRESHOLD, // threshold
            0.001,                      // weak L2 (clean synthetic data)
            0.1,                        // learning rate
            500,                        // max iter
            None,                       // rng_seed (entropy-based)
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
        let pos_seq = make_seq(positives, crate::embedding_sequence::LabelStratum::Positive);
        let neg_seq = make_seq(negatives, crate::embedding_sequence::LabelStratum::Negative);

        let verifier = VoiceVerifier::train(
            &[pos_seq],
            &[neg_seq],
            None,                       // no per-negative weights
            DEFAULT_VERIFIER_THRESHOLD, // mahbot-853: lowered from 0.6 for streaming inference.
            L2_LAMBDA,                  // L2 regularization (mahbot-854: 0.01)
            LEARNING_RATE,              // learning rate (mahbot-878: 0.001, Adam)
            MAX_ITER,                   // max iterations (mahbot-854: 5000)
            None,                       // rng_seed (entropy-based)
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
        let pos_seq = make_seq(positives, crate::embedding_sequence::LabelStratum::Positive);

        let verifier = VoiceVerifier::train_with_synthetic_negatives(
            &[pos_seq],
            DEFAULT_VERIFIER_THRESHOLD,
            None,
        );

        assert!(verifier.is_trained(), "Verifier must be trained");
        assert_eq!(
            verifier.threshold, DEFAULT_VERIFIER_THRESHOLD,
            "threshold must match DEFAULT_VERIFIER_THRESHOLD",
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
    fn test_verifier_serialization_roundtrip() {
        // Train a verifier.
        let mut rng = StdRng::seed_from_u64(42);
        let positives: Vec<Vec<f32>> = (0..10).map(|_| make_positive_embedding(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..10).map(|_| make_negative_embedding(&mut rng)).collect();
        let pos_seq = make_seq(positives, crate::embedding_sequence::LabelStratum::Positive);
        let neg_seq = make_seq(negatives, crate::embedding_sequence::LabelStratum::Negative);

        let verifier = VoiceVerifier::train(
            &[pos_seq],
            &[neg_seq],
            None, // no per-negative weights
            DEFAULT_VERIFIER_THRESHOLD,
            0.001,
            0.1,
            500,
            None, // rng_seed (entropy-based)
        );

        // Serialize to JSON.
        let json = serde_json::to_string(&verifier).expect("serialize");

        // Deserialize.
        let deserialized: VoiceVerifier = serde_json::from_str(&json).expect("deserialize");

        // Verify same predictions on held-out test vectors.
        let test_pos = make_positive_embedding(&mut rng);
        let test_neg = make_negative_embedding(&mut rng);

        let score_before = verifier.predict(&test_pos);
        let score_after = deserialized.predict(&test_pos);
        assert!(
            (score_before - score_after).abs() < 1e-4,
            "Positive prediction must match after roundtrip: before={score_before:.4} after={score_after:.4}",
        );

        let score_before = verifier.predict(&test_neg);
        let score_after = deserialized.predict(&test_neg);
        assert!(
            (score_before - score_after).abs() < 1e-4,
            "Negative prediction must match after roundtrip: before={score_before:.4} after={score_after:.4}",
        );
    }

    #[test]
    fn test_logistic_verifier_serialization_roundtrip() {
        // Train a logistic model and verify JSON roundtrip preserves predictions,
        // verifier_version, and is_trained() status.
        let mut rng = StdRng::seed_from_u64(42);
        let positives: Vec<Vec<f32>> = (0..30).map(|_| make_positive_frame(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..50).map(|_| make_negative_frame(&mut rng)).collect();

        // L2-normalize the training features (matching training pipeline ordering).
        let features: Vec<Vec<f32>> = positives.iter().chain(negatives.iter()).cloned().collect();
        let normalized: Vec<Vec<f32>> = features
            .iter()
            .map(|f| {
                let norm = f.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
                f.iter().map(|v| v / norm).collect()
            })
            .collect();

        // Compute StandardScaler on L2-normalized features.
        let (scaler_mean, scaler_std) = compute_standard_scaler(&normalized);

        // Scale the L2-normalized features for training.
        let scaled: Vec<Vec<f32>> = normalized
            .iter()
            .map(|f| {
                f.iter()
                    .enumerate()
                    .map(|(j, &val)| {
                        if scaler_std[j] > 0.0 {
                            (val - scaler_mean[j]) / scaler_std[j]
                        } else {
                            val
                        }
                    })
                    .collect()
            })
            .collect();

        let labels: Vec<f32> = [vec![1.0; 30], vec![0.0; 50]].concat();
        let sample_weights: Vec<f32> = [vec![3.0; 30], vec![1.0; 50]].concat();

        let (weights, bias) =
            train_logistic_sgd(&scaled, &labels, &sample_weights, 0.01, 0.01, 500, Some(42));

        // Build a logistic verifier as train() would.
        let verifier = VoiceVerifier {
            weights,
            bias,
            w1: Vec::new(),
            b1: Vec::new(),
            w2: Vec::new(),
            b2: Vec::new(),
            w3: Vec::new(),
            b3: 0.0,
            scaler_mean,
            scaler_std,
            threshold: DEFAULT_VERIFIER_THRESHOLD,
            trained: true,
            verifier_version: VERIFIER_VERSION_LOGISTIC,
        };

        // Verify it's considered trained.
        assert!(verifier.is_trained());

        // Serialize to JSON.
        let json = serde_json::to_string(&verifier).expect("serialize");

        // Verify JSON contains the version field.
        assert!(
            json.contains("\"verifier_version\":1"),
            "JSON should contain verifier_version=1, got: {}",
            &json[..json.len().min(200)],
        );

        // Deserialize.
        let deserialized: VoiceVerifier = serde_json::from_str(&json).expect("deserialize");

        // Verify version is preserved.
        assert_eq!(deserialized.verifier_version, VERIFIER_VERSION_LOGISTIC);

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
    }

    #[test]
    fn test_logistic_train_inner_end_to_end() {
        // End-to-end test for the full logistic training pipeline via train_inner:
        // form_stride1_pooled_windows → scaler fitting → train_logistic_sgd → model
        // construction.  Unlike test_logistic_verifier_serialization_roundtrip (which
        // manually constructs the model), this exercises the actual train_inner path.
        let mut rng = StdRng::seed_from_u64(42);
        let positives: Vec<Vec<f32>> = (0..30).map(|_| make_positive_frame(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..50).map(|_| make_negative_frame(&mut rng)).collect();
        let pos_seq = make_seq(positives, crate::embedding_sequence::LabelStratum::Positive);
        let neg_seq = make_seq(negatives, crate::embedding_sequence::LabelStratum::Negative);

        let verifier = VoiceVerifier::train_inner(
            &[pos_seq],
            &[neg_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            L2_LAMBDA,
            LOGISTIC_LEARNING_RATE,
            LOGISTIC_MAX_ITER,
            Some(42),
            true, // use_logistic
        );

        assert!(verifier.is_trained());
        assert_eq!(
            verifier.verifier_version, VERIFIER_VERSION_LOGISTIC,
            "train_inner(use_logistic=true) must produce version LOGISTIC",
        );
        assert_eq!(
            verifier.weights.len(),
            EMBEDDING_DIM,
            "logistic verifier weights must be 96-dim",
        );
        assert!(verifier.bias.is_finite(), "bias must be finite",);
        assert_eq!(
            verifier.scaler_mean.len(),
            EMBEDDING_DIM,
            "scaler_mean must be 96-dim",
        );
        assert_eq!(
            verifier.scaler_std.len(),
            EMBEDDING_DIM,
            "scaler_std must be 96-dim",
        );

        // Verify discrimination on held-out 96-dim per-frame input.
        let held_out_pos = make_positive_frame(&mut rng);
        let held_out_neg = make_negative_frame(&mut rng);
        let score_pos = verifier.predict(&held_out_pos);
        let score_neg = verifier.predict(&held_out_neg);
        assert!(
            score_pos > score_neg,
            "Logistic verifier must discriminate: pos={score_pos:.4} neg={score_neg:.4}",
        );
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
    }

    #[test]
    fn test_mean_pool_empty() {
        let pooled = mean_pool_embeddings(&[]);
        assert!(pooled.is_empty());
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
    }

    #[test]
    fn test_generate_synthetic_negatives_zero_count() {
        let negs = generate_synthetic_negatives(0, 96);
        assert!(negs.is_empty());
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
    }

    #[test]
    fn test_generate_synthetic_negatives_from_positives_zero_count() {
        let positives: Vec<Vec<f32>> = vec![vec![0.5; 96]];
        let negs = generate_synthetic_negatives_from_positives(0, &positives, 1.5, None);
        assert!(negs.is_empty());
    }

    #[test]
    fn test_generate_synthetic_negatives_from_positives_empty_positives() {
        let positives: Vec<Vec<f32>> = vec![];
        let negs = generate_synthetic_negatives_from_positives(10, &positives, 1.5, None);
        assert!(negs.is_empty());
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
    }

    #[test]
    fn test_compute_standard_scaler_empty() {
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
            verifier_version: VERIFIER_VERSION_LEGACY,
            weights: vec![0.5; VERIFIER_INPUT_DIM],
            bias: 0.0,
            w1: Vec::new(),
            b1: Vec::new(),
            w2: Vec::new(),
            b2: Vec::new(),
            w3: Vec::new(),
            b3: 0.0,
            scaler_mean: vec![0.1; 96], // wrong dimension (96 ≠ 288)
            scaler_std: vec![0.2; 96],
            threshold: DEFAULT_VERIFIER_THRESHOLD,
        };
        assert!(
            !verifier.is_trained(),
            "Mismatched scaler dims should report untrained"
        );

        // Also test partial mismatch: only scaler_std populated.
        let verifier2 = VoiceVerifier {
            trained: true,
            verifier_version: VERIFIER_VERSION_LEGACY,
            weights: vec![0.5; VERIFIER_INPUT_DIM],
            bias: 0.0,
            w1: Vec::new(),
            b1: Vec::new(),
            w2: Vec::new(),
            b2: Vec::new(),
            w3: Vec::new(),
            b3: 0.0,
            scaler_mean: Vec::new(),
            scaler_std: vec![0.2; 96], // non-empty but mismatched
            threshold: DEFAULT_VERIFIER_THRESHOLD,
        };
        assert!(
            !verifier2.is_trained(),
            "Partial mismatched scaler should report untrained"
        );
    }

    #[test]
    fn test_verifier_noop_untrained_serialization() {
        // Serialize and deserialize an untrained verifier — must remain no-op.
        let verifier = VoiceVerifier::untrained();
        let json = serde_json::to_string(&verifier).expect("serialize");
        let deserialized: VoiceVerifier = serde_json::from_str(&json).expect("deserialize");

        assert!(!deserialized.is_trained());
        let score = deserialized.predict(&[0.0; VERIFIER_INPUT_DIM]);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_train_with_synthetic_negatives_basic() {
        let mut rng = StdRng::seed_from_u64(42);
        let positives: Vec<Vec<f32>> = (0..30).map(|_| make_positive_embedding(&mut rng)).collect();
        let pos_seq = make_seq(positives, crate::embedding_sequence::LabelStratum::Positive);
        let verifier = VoiceVerifier::train_with_synthetic_negatives(
            &[pos_seq],
            DEFAULT_VERIFIER_THRESHOLD,
            None,
        );
        assert!(verifier.is_trained());
        assert_eq!(
            verifier.threshold, DEFAULT_VERIFIER_THRESHOLD,
            "threshold must match DEFAULT_VERIFIER_THRESHOLD",
        );
        // These MLP-specific field assertions only apply when training produced
        // an MLP model (the default). When MAHBOT_USE_LOGISTIC_VERIFIER is set,
        // the model has weights at EMBEDDING_DIM (96) instead.
        if !use_logistic_verifier() {
            assert_eq!(verifier.w1.len(), VERIFIER_INPUT_DIM * MLP_HIDDEN_1);
            assert_eq!(verifier.b1.len(), MLP_HIDDEN_1);
        }
        assert!(!verifier.scaler_mean.is_empty());
        assert!(!verifier.scaler_std.is_empty());

        // All MLP weights must be finite — NaN/inf indicates gradient divergence
        // from unstable hyperparameters.  Only relevant for MLP models.
        if !use_logistic_verifier() {
            for (j, &w) in verifier.w1.iter().enumerate() {
                assert!(
                    w.is_finite(),
                    "w1[{j}] is not finite: {w}; gradient descent diverged",
                );
            }
            for (j, &w) in verifier.w2.iter().enumerate() {
                assert!(
                    w.is_finite(),
                    "w2[{j}] is not finite: {w}; gradient descent diverged",
                );
            }
            for (j, &w) in verifier.w3.iter().enumerate() {
                assert!(
                    w.is_finite(),
                    "w3[{j}] is not finite: {w}; gradient descent diverged",
                );
            }
        }

        // Predict must return a reasonable score for a positive embedding.
        let held_out = make_positive_embedding(&mut rng);
        let score = verifier.predict(&held_out);
        assert!(
            score >= 0.5,
            "Verifier should accept positive embedding (score >= 0.5), got score={score:.4}; \
             weights may have diverged",
        );
    }

    #[test]
    fn test_backward_compat_legacy_linear_model() {
        // Legacy logistic regression models (pre-mahbot-861) used weights+bias
        // with optional StandardScaler.  This test verifies that such models
        // deserialize correctly and produce correct predictions through the
        // linear fallback path in predict().

        // Known 288-dim input.
        let input: Vec<f32> = (0..VERIFIER_INPUT_DIM).map(|i| i as f32 * 0.01).collect();
        // Known weights and bias.
        let weights: Vec<f32> = (0..VERIFIER_INPUT_DIM)
            .map(|i| (i % 3) as f32 * 0.1 - 0.2)
            .collect();
        let bias = 0.25;

        // ── Without scaler ────────────────────────────────────────────
        let verifier = VoiceVerifier {
            trained: true,
            verifier_version: VERIFIER_VERSION_LEGACY,
            weights: weights.clone(),
            bias,
            w1: Vec::new(),
            b1: Vec::new(),
            w2: Vec::new(),
            b2: Vec::new(),
            w3: Vec::new(),
            b3: 0.0,
            scaler_mean: Vec::new(),
            scaler_std: Vec::new(),
            threshold: DEFAULT_VERIFIER_THRESHOLD,
        };
        assert!(
            verifier.is_trained(),
            "Legacy linear verifier (no scaler) must report trained",
        );

        // Compute expected: sigmoid(dot(input, weights) + bias).
        let dot: f32 = input.iter().zip(weights.iter()).map(|(x, w)| x * w).sum();
        let expected = sigmoid(dot + bias);
        let actual = verifier.predict(&input);
        assert!(
            (actual - expected).abs() < 1e-5,
            "Legacy linear (no scaler): expected {expected:.6}, got {actual:.6}",
        );

        // ── With scaler (rejected) ─────────────────────────────────────
        // Linear + scaler is not a valid configuration (mahbot-870):
        // predict() ignores the scaler for linear models (legacy linear
        // models predate both L2-normalization and the StandardScaler), so
        // accepting this combination would silently produce wrong predictions.
        // is_trained() correctly rejects it.
        let scaler_mean: Vec<f32> = (0..VERIFIER_INPUT_DIM)
            .map(|i| (i as f32).sin() * 0.1)
            .collect();
        let scaler_std: Vec<f32> = (0..VERIFIER_INPUT_DIM)
            .map(|i| (i as f32).cos().abs() + 0.1)
            .collect();
        let verifier_scaled = VoiceVerifier {
            trained: true,
            verifier_version: VERIFIER_VERSION_LEGACY,
            weights: weights.clone(),
            bias,
            w1: Vec::new(),
            b1: Vec::new(),
            w2: Vec::new(),
            b2: Vec::new(),
            w3: Vec::new(),
            b3: 0.0,
            scaler_mean,
            scaler_std,
            threshold: DEFAULT_VERIFIER_THRESHOLD,
        };
        assert!(
            !verifier_scaled.is_trained(),
            "Legacy linear verifier (with scaler) must report untrained — \
             predict() ignores the scaler for linear models (mahbot-870)",
        );
    }

    #[test]
    fn test_backward_compat_linear_wrong_weights_dim_detected() {
        // A legacy linear verifier with wrong-length weights must be rejected
        // by is_trained() — wrong dimensions would produce truncated dot
        // products via zip() in predict().
        let verifier = VoiceVerifier {
            trained: true,
            verifier_version: VERIFIER_VERSION_LEGACY,
            weights: vec![0.5; 96], // wrong: 96 ≠ 288
            bias: 0.0,
            w1: Vec::new(),
            b1: Vec::new(),
            w2: Vec::new(),
            b2: Vec::new(),
            w3: Vec::new(),
            b3: 0.0,
            scaler_mean: Vec::new(),
            scaler_std: Vec::new(),
            threshold: DEFAULT_VERIFIER_THRESHOLD,
        };
        assert!(
            !verifier.is_trained(),
            "Legacy linear verifier with wrong weights dim (96) must report untrained",
        );

        // Also verify that correct-length weights still passes.
        let verifier_ok = VoiceVerifier {
            weights: vec![0.5; VERIFIER_INPUT_DIM],
            ..verifier
        };
        assert!(
            verifier_ok.is_trained(),
            "Legacy linear verifier with correct weights dim (288) must report trained",
        );
    }

    #[test]
    fn test_backward_compat_corrupted_mlp_falls_back_to_linear() {
        // When MLP parameters exist but have wrong dimensions (corrupted
        // serialization), is_trained() should accept the linear fallback
        // and predict() should use weights/bias instead of mlp_forward().

        let input: Vec<f32> = (0..VERIFIER_INPUT_DIM).map(|i| i as f32 * 0.01).collect();

        // Helper to create alternating-sign weights (avoids activation
        // saturation from uniform positive weights).
        let alternating = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|i| if i % 2 == 0 { 0.2 } else { -0.2 })
                .collect()
        };

        // Corrupted MLP: w1 has wrong length (half the correct size).
        let verifier = VoiceVerifier {
            trained: true,
            verifier_version: VERIFIER_VERSION_LEGACY,
            weights: alternating(VERIFIER_INPUT_DIM), // valid linear weights
            bias: 0.5,
            w1: alternating(VERIFIER_INPUT_DIM * MLP_HIDDEN_1 / 2), // wrong length
            b1: alternating(MLP_HIDDEN_1),
            w2: alternating(MLP_HIDDEN_1 * MLP_HIDDEN_2),
            b2: alternating(MLP_HIDDEN_2),
            w3: alternating(MLP_HIDDEN_2),
            b3: 0.7,
            scaler_mean: Vec::new(),
            scaler_std: Vec::new(),
            threshold: DEFAULT_VERIFIER_THRESHOLD,
        };

        assert!(
            verifier.is_trained(),
            "Must report trained when MLP is corrupted but linear is valid",
        );

        // predict() must use linear path, producing sigmoid(dot(input, weights) + bias).
        // Note: linear path uses raw (non-L2-normalized) input for backward compat.
        let dot: f32 = input
            .iter()
            .zip(verifier.weights.iter())
            .map(|(x, w)| x * w)
            .sum();
        let expected = sigmoid(dot + verifier.bias);
        let actual = verifier.predict(&input);
        assert!(
            (actual - expected).abs() < 1e-5,
            "Corrupted MLP + valid linear: expected {expected:.6} (linear), got {actual:.6}",
        );

        // Also verify that a completely valid verifier (both MLP and linear valid)
        // uses the MLP path (MLP takes priority).
        let valid_verifier = VoiceVerifier {
            w1: alternating(VERIFIER_INPUT_DIM * MLP_HIDDEN_1), // now correct length
            ..verifier
        };
        assert!(
            valid_verifier.is_trained(),
            "Valid MLP + valid linear must report trained",
        );
        // MLP path should produce a different score than linear path.
        let mlp_score = valid_verifier.predict(&input);
        assert!(
            (mlp_score - expected).abs() > 1e-4,
            "Valid MLP must not silently use linear fallback — MLP score {mlp_score:.6} \
             should differ from linear score {expected:.6}",
        );
    }

    #[test]
    fn test_mlp_consistency_predict_matches_mlp_forward() {
        // Verify that predict() and mlp_forward() produce identical results
        // when called on the same input with the same MLP parameters.
        let mut rng = StdRng::seed_from_u64(42);
        let positives: Vec<Vec<f32>> = (0..20).map(|_| make_positive_embedding(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..30).map(|_| make_negative_embedding(&mut rng)).collect();
        let pos_seq = make_seq(positives, crate::embedding_sequence::LabelStratum::Positive);
        let neg_seq = make_seq(negatives, crate::embedding_sequence::LabelStratum::Negative);

        let verifier = VoiceVerifier::train(
            &[pos_seq],
            &[neg_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            0.001,
            0.1,
            500,
            None, // rng_seed (entropy-based)
        );
        assert!(
            verifier.is_trained(),
            "Verifier must be trained for consistency check",
        );

        // Test on several held-out embeddings.
        for _ in 0..20 {
            let emb = if rng.random::<f32>() < 0.5 {
                make_positive_embedding(&mut rng)
            } else {
                make_negative_embedding(&mut rng)
            };

            // Apply the same pipeline as predict(): L2-norm → scaler.
            // First L2-normalize (matching training convention).
            let norm_l2: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
            let emb_l2: Vec<f32> = emb.iter().map(|v| v / norm_l2).collect();
            // Then apply scaler to get the fully processed input.
            let x: Vec<f32> = if !verifier.scaler_mean.is_empty() && !verifier.scaler_std.is_empty()
            {
                emb_l2
                    .iter()
                    .zip(verifier.scaler_mean.iter())
                    .zip(verifier.scaler_std.iter())
                    .map(
                        |((&val, &mean), &std)| {
                            if std > 0.0 { (val - mean) / std } else { val }
                        },
                    )
                    .collect()
            } else {
                emb_l2 // no scaler — L2-normalized input is the fully processed form
            };

            // predict() applies L2-norm → scaler internally + routes to MLP.
            // predict_scaled() skips preprocessing + routes to MLP directly.
            // Both must produce the same output (pipeline consistency).
            let from_predict = verifier.predict(&emb);
            let from_scaled = predict_scaled(&verifier, &x);
            assert!(
                (from_predict - from_scaled).abs() < 1e-5,
                "predict()={from_predict:.6} ≠ predict_scaled()={from_scaled:.6} \
                 — pipeline routing mismatch",
            );

            // predict_scaled() must match direct mlp_forward() call
            // (routing correctness, not a tautology since predict_scaled
            //  uses the same has_valid_mlp_params() logic as predict()).
            // Input is already fully processed (L2-norm → scaler), so
            // mlp_forward receives x directly (mahbot-870).
            let expected = mlp_forward(
                &x,
                &verifier.w1,
                &verifier.b1,
                &verifier.w2,
                &verifier.b2,
                &verifier.w3,
                verifier.b3,
            );
            assert!(
                (from_scaled - expected).abs() < 1e-5,
                "predict_scaled()={from_scaled:.6} ≠ mlp_forward()={expected:.6} \
                 — routing mismatch: w1.len={}, b1.len={}, w2.len={}, b2.len={}, \
                 w3.len={}, b3={}",
                verifier.w1.len(),
                verifier.b1.len(),
                verifier.w2.len(),
                verifier.b2.len(),
                verifier.w3.len(),
                verifier.b3,
            );
        }
    }

    #[test]
    fn test_verifier_empty_training_returns_untrained() {
        // No positive examples → should return untrained.
        let neg_embs = vec![vec![0.0; VERIFIER_INPUT_DIM]];
        let neg_seq = make_seq(neg_embs, crate::embedding_sequence::LabelStratum::Negative);
        let verifier = VoiceVerifier::train(
            &[],
            &[neg_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            0.001,
            0.1,
            100,
            None, // rng_seed (entropy-based)
        );
        assert!(!verifier.is_trained());
    }

    #[test]
    fn test_deterministic_training_same_seed_identical_weights() {
        // Two training runs with the same seed and identical training data
        // must produce identical MLP weights.
        let mut rng = StdRng::seed_from_u64(12345);
        let positives: Vec<Vec<f32>> = (0..10).map(|_| make_positive_embedding(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..10).map(|_| make_negative_embedding(&mut rng)).collect();
        let pos_seq = make_seq(positives, crate::embedding_sequence::LabelStratum::Positive);
        let neg_seq = make_seq(negatives, crate::embedding_sequence::LabelStratum::Negative);

        let seed = 42;
        let v1 = VoiceVerifier::train(
            &[pos_seq.clone()],
            &[neg_seq.clone()],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            0.001,
            0.1,
            100,
            Some(seed),
        );
        let v2 = VoiceVerifier::train(
            &[pos_seq],
            &[neg_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            0.001,
            0.1,
            100,
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
            v1.w1, v2.w1,
            "w1 differs between deterministic training runs"
        );
        assert_eq!(
            v1.w2, v2.w2,
            "w2 differs between deterministic training runs"
        );
        assert_eq!(
            v1.w3, v2.w3,
            "w3 differs between deterministic training runs"
        );
        assert_eq!(
            v1.b1, v2.b1,
            "b1 differs between deterministic training runs"
        );
        assert_eq!(
            v1.b2, v2.b2,
            "b2 differs between deterministic training runs"
        );
        assert_eq!(
            v1.b3, v2.b3,
            "b3 differs between deterministic training runs"
        );
    }

    #[test]
    fn test_form_stride1_windows_basic() {
        // Verify that form_stride1_windows produces correctly shaped
        // L2-normalized windows from 96-dim per-frame embeddings.
        let n_frames = 5;
        let embeddings: Vec<Vec<f32>> = (0..n_frames)
            .map(|i| vec![i as f32; EMBEDDING_DIM])
            .collect();

        let windows = form_stride1_windows(&embeddings);
        // 5 frames → 3 stride-1 windows (frames 0-2, 1-3, 2-4).
        assert_eq!(windows.len(), n_frames - VERIFIER_WINDOW_SIZE + 1);
        for w in &windows {
            assert_eq!(w.len(), VERIFIER_INPUT_DIM);
            // Verify L2-normalized (norm ≈ 1.0).
            let norm: f32 = w.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "window norm={norm} ≠ 1.0");
        }
    }

    #[test]
    fn test_form_stride1_windows_short_input() {
        // Fewer than VERIFIER_WINDOW_SIZE frames → empty result.
        let embeddings: Vec<Vec<f32>> = (0..2).map(|i| vec![i as f32; EMBEDDING_DIM]).collect();
        let windows = form_stride1_windows(&embeddings);
        assert!(windows.is_empty());
    }

    #[test]
    fn test_form_stride1_windows_empty_input() {
        let windows = form_stride1_windows(&[]);
        assert!(windows.is_empty());
    }

    // ── Logistic verifier tests (mahbot-901) ──────────────────────────

    /// Generate a synthetic 96-dim per-frame embedding with values clustered
    /// around +0.5 (simulates positive wake-word frame).
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

    #[test]
    fn test_logistic_sgd_train_and_predict() {
        // Train logistic SGD on 96-dim per-frame positive/negative embeddings,
        // then verify prediction on held-out data discriminates correctly.
        let mut rng = StdRng::seed_from_u64(42);
        let positives: Vec<Vec<f32>> = (0..30).map(|_| make_positive_frame(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..50).map(|_| make_negative_frame(&mut rng)).collect();

        let features: Vec<Vec<f32>> = positives.iter().chain(negatives.iter()).cloned().collect();
        let labels: Vec<f32> = [vec![1.0; 30], vec![0.0; 50]].concat();
        // Class-weight positives to compensate for imbalance (50 neg / 30 pos).
        let sample_weights: Vec<f32> = [vec![50.0 / 30.0; 30], vec![1.0; 50]].concat();

        let (weights, bias) = train_logistic_sgd(
            &features,
            &labels,
            &sample_weights,
            L2_LAMBDA,              // 0.01 — production L2 regularisation
            LOGISTIC_LEARNING_RATE, // 0.01 — production learning rate for logistic SGD
            LOGISTIC_MAX_ITER,      // 1000 — production max iterations for logistic
            Some(42),               // deterministic seed
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
        let held_out_pos: Vec<f32> = make_positive_frame(&mut rng);
        let held_out_neg: Vec<f32> = make_negative_frame(&mut rng);

        // Use predict_logistic() directly (no scaler fitted in this test).
        let score_pos = predict_logistic(&held_out_pos, &weights, bias, &[], &[]);
        let score_neg = predict_logistic(&held_out_neg, &weights, bias, &[], &[]);
        assert!(
            score_pos > score_neg,
            "Logistic should score positive ({score_pos:.4}) higher than negative ({score_neg:.4})",
        );
    }

    #[test]
    fn test_logistic_sgd_deterministic() {
        // Two training runs with the same seed must produce identical weights.
        let mut rng = StdRng::seed_from_u64(12345);
        let positives: Vec<Vec<f32>> = (0..10).map(|_| make_positive_frame(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..10).map(|_| make_negative_frame(&mut rng)).collect();
        let features: Vec<Vec<f32>> = positives.iter().chain(negatives.iter()).cloned().collect();
        let labels: Vec<f32> = [vec![1.0; 10], vec![0.0; 10]].concat();
        let sample_weights: Vec<f32> = [vec![10.0; 10], vec![1.0; 10]].concat(); // class-weighted

        let seed = 42;
        let (w1, b1) = train_logistic_sgd(
            &features,
            &labels,
            &sample_weights,
            L2_LAMBDA,
            LOGISTIC_LEARNING_RATE,
            LOGISTIC_MAX_ITER,
            Some(seed),
        );
        let (w2, b2) = train_logistic_sgd(
            &features,
            &labels,
            &sample_weights,
            L2_LAMBDA,
            LOGISTIC_LEARNING_RATE,
            LOGISTIC_MAX_ITER,
            Some(seed),
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

    #[test]
    fn test_mean_pool_window_into_basic() {
        // Mean-pool a simple 3-frame pattern and verify the output.
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
            // For dim 0: (0+10+20)/3 = 10; dim 1: (1+11+21)/3 = 11; etc.
            let correct = ((i + 0) + (i + 10) + (i + 20)) as f32 / 3.0;
            assert!(
                (pooled[i] - correct).abs() < 1e-5,
                "pooled[{i}] = {}, expected {correct}",
                pooled[i],
            );
        }
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
        let pos_seq = make_seq(embs1, crate::embedding_sequence::LabelStratum::Positive);
        let neg_seq = make_seq(embs2, crate::embedding_sequence::LabelStratum::Negative);

        // With per-sequence windowing, each sequence has 2 frames < 3 → 0 windows each
        // → train_inner gets 0 positive windows + 0 negative windows → untrained.
        let verifier = VoiceVerifier::train(
            &[pos_seq],
            &[neg_seq],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            L2_LAMBDA,
            LEARNING_RATE,
            MAX_ITER,
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
            make_seq(pos1, crate::embedding_sequence::LabelStratum::Positive),
            make_seq(pos2, crate::embedding_sequence::LabelStratum::Positive),
        ];
        let neg_seqs = [
            make_seq(neg1, crate::embedding_sequence::LabelStratum::Negative),
            make_seq(neg2, crate::embedding_sequence::LabelStratum::Negative),
        ];

        let verifier = VoiceVerifier::train(
            &pos_seqs,
            &neg_seqs,
            None, // no per-negative weights
            DEFAULT_VERIFIER_THRESHOLD,
            L2_LAMBDA,
            LEARNING_RATE,
            MAX_ITER,
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
    }

    #[test]
    fn test_verifier_train_with_cache_sequences() {
        // Simulates production cache path: confusable + unrelated + synthetic
        // negatives as separate sequences with per-sequence weight tiers.
        let mut rng = StdRng::seed_from_u64(42);
        let positives: Vec<Vec<f32>> = (0..20).map(|_| make_positive_embedding(&mut rng)).collect();
        let pos_seq = make_seq(positives, crate::embedding_sequence::LabelStratum::Positive);

        // Three negative sequences simulating confusable, unrelated, synthetic
        let neg_confusable: Vec<Vec<f32>> =
            (0..10).map(|_| make_negative_embedding(&mut rng)).collect();
        let neg_unrelated: Vec<Vec<f32>> =
            (0..10).map(|_| make_negative_embedding(&mut rng)).collect();
        let neg_synthetic: Vec<Vec<f32>> =
            (0..10).map(|_| make_negative_embedding(&mut rng)).collect();

        let conf_seq = make_seq(
            neg_confusable,
            crate::embedding_sequence::LabelStratum::Negative,
        );
        let unrel_seq = make_seq(
            neg_unrelated,
            crate::embedding_sequence::LabelStratum::Negative,
        );
        let synth_seq = make_seq(
            neg_synthetic,
            crate::embedding_sequence::LabelStratum::Negative,
        );

        // Per-sequence weights: confusable=3.0, unrelated=2.0, synthetic=1.0
        let per_neg_weights = vec![3.0, 2.0, 1.0];

        let verifier = VoiceVerifier::train(
            &[pos_seq],
            &[conf_seq, unrel_seq, synth_seq],
            Some(&per_neg_weights),
            DEFAULT_VERIFIER_THRESHOLD,
            L2_LAMBDA,
            LEARNING_RATE,
            MAX_ITER,
            Some(42),
        );

        assert!(verifier.is_trained());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Hard-negative mining tests (mahbot-905)
    // ═══════════════════════════════════════════════════════════════════════

    /// Helper: create an EmbeddingSequence from 96-dim per-frame embeddings
    /// with a specific Source for provenance tracking.
    fn make_seq_with_source(
        embs: Vec<Vec<f32>>,
        source: crate::embedding_sequence::Source,
    ) -> EmbeddingSequence {
        EmbeddingSequence {
            id: crate::embedding_sequence::UtteranceId {
                sequence_index: 0,
                variant_index: 0,
            },
            source,
            augmentation_family: None,
            label_stratum: crate::embedding_sequence::LabelStratum::Negative,
            embeddings: embs,
        }
    }

    /// Create a deterministic classifier for mining tests using a seeded RNG.
    fn test_classifier() -> WakeWordClassifier {
        use crate::wake_word_classifier::{ArchConfig, ClassifierWeights};
        let weights =
            ClassifierWeights::from_rng(&mut StdRng::seed_from_u64(42), &ArchConfig::default());
        WakeWordClassifier::new(weights)
    }

    #[test]
    fn test_mine_hard_negatives_empty_input() {
        // Empty input slice → empty MinedNegatives.
        let classifier = test_classifier();
        let mined = mine_hard_negatives(&classifier, &[], 2, VERIFIER_WINDOW_SIZE);
        assert!(
            mined.sequences.is_empty(),
            "Empty input must produce empty mined sequences",
        );
        assert_eq!(
            mined.source_sequences_represented, 0,
            "Empty input must have zero source_sequences_represented",
        );
    }

    #[test]
    fn test_mine_hard_negatives_short_sequences() {
        // All sequences shorter than VERIFIER_WINDOW_SIZE → empty result.
        let classifier = test_classifier();
        let seq1 = make_seq_with_source(
            (0..2)
                .map(|_| make_positive_frame(&mut StdRng::seed_from_u64(10)))
                .collect(),
            crate::embedding_sequence::Source::Confusable,
        );
        let mined = mine_hard_negatives(&classifier, &[seq1], 2, VERIFIER_WINDOW_SIZE);
        assert!(
            mined.sequences.is_empty(),
            "Sequences shorter than window size must produce no mined windows",
        );
    }

    #[test]
    fn test_mine_hard_negatives_basic() {
        // Normal sequences → returns sequences with correct properties.
        let classifier = test_classifier();
        let mut rng = StdRng::seed_from_u64(100);

        // Three sequences with varying frame counts (all ≥ VERIFIER_WINDOW_SIZE).
        let seq1 = make_seq_with_source(
            (0..10).map(|_| make_positive_frame(&mut rng)).collect(),
            crate::embedding_sequence::Source::Confusable,
        );
        let seq2 = make_seq_with_source(
            (0..8).map(|_| make_positive_frame(&mut rng)).collect(),
            crate::embedding_sequence::Source::Unrelated,
        );
        let seq3 = make_seq_with_source(
            (0..6).map(|_| make_positive_frame(&mut rng)).collect(),
            crate::embedding_sequence::Source::Ambient,
        );

        let mined = mine_hard_negatives(
            &classifier,
            &[seq1, seq2, seq3],
            2, // max_per_sequence
            VERIFIER_WINDOW_SIZE,
        );

        // At most 2 per sequence × 3 sequences = 6 windows.
        assert!(
            mined.sequences.len() <= 6,
            "At most 2 per sequence × 3 sequences = 6 windows, got {}",
            mined.sequences.len(),
        );
        assert!(
            mined.source_sequences_represented <= 3,
            "At most 3 source sequences represented",
        );

        // Each mined sequence must have exactly VERIFIER_WINDOW_SIZE
        // per-frame 96-dim embeddings.
        for seq in &mined.sequences {
            assert_eq!(
                seq.embeddings.len(),
                VERIFIER_WINDOW_SIZE,
                "Each mined sequence must have exactly {VERIFIER_WINDOW_SIZE} embeddings",
            );
            for emb in &seq.embeddings {
                assert_eq!(
                    emb.len(),
                    EMBEDDING_DIM,
                    "Each embedding must be {EMBEDDING_DIM}-dim",
                );
            }
            // Provenance must be preserved.
            assert_eq!(
                seq.label_stratum,
                crate::embedding_sequence::LabelStratum::Negative,
                "Mined sequences must remain negative",
            );
        }
    }

    #[test]
    fn test_mine_hard_negatives_max_per_sequence_enforced() {
        // Verify that at most max_per_sequence windows are selected
        // from each source sequence.
        let classifier = test_classifier();
        let mut rng = StdRng::seed_from_u64(200);

        // A single long sequence with many frames.
        let seq = make_seq_with_source(
            (0..50).map(|_| make_positive_frame(&mut rng)).collect(),
            crate::embedding_sequence::Source::Confusable,
        );

        let mined = mine_hard_negatives(
            &classifier,
            &[seq],
            2, // max_per_sequence
            VERIFIER_WINDOW_SIZE,
        );

        // At most 2 windows from a single sequence.
        assert!(
            mined.sequences.len() <= 2,
            "At most 2 windows per sequence, got {}",
            mined.sequences.len(),
        );
        assert_eq!(
            mined.source_sequences_represented, 1,
            "One source sequence should be represented",
        );
    }

    #[test]
    fn test_mine_hard_negatives_non_overlapping() {
        // Verify that selected windows from the same source sequence
        // are non-overlapping (separation ≥ VERIFIER_WINDOW_SIZE).
        let classifier = test_classifier();

        // Create embeddings where each frame has a unique value pattern:
        // frame i has all 96 values = i (as f32).  This lets us identify
        // which frame index a window's embeddings come from by examining any
        // single element (e.g., the first element of each 96-dim embedding).
        let n_frames = 30;
        let embs: Vec<Vec<f32>> = (0..n_frames)
            .map(|i| vec![i as f32; EMBEDDING_DIM])
            .collect();
        let seq = make_seq_with_source(embs, crate::embedding_sequence::Source::Confusable);

        let mined = mine_hard_negatives(
            &classifier,
            &[seq],
            4, // max_per_sequence = 4 to test with more windows
            VERIFIER_WINDOW_SIZE,
        );

        // Each mined window is a 3-frame EmbeddingSequence.  Extract the
        // frame indices from each window (via the first element of each of
        // its 3 embeddings, which equals the original frame index).
        let window_frames: Vec<Vec<usize>> = mined
            .sequences
            .iter()
            .map(|s| {
                s.embeddings
                    .iter()
                    .map(|e| e.first().copied().unwrap_or(0.0) as usize)
                    .collect()
            })
            .collect();

        // Verify pairwise non-overlap: any two windows from the same source
        // must not share any frame index.
        for (i, a_frames) in window_frames.iter().enumerate() {
            for (j, b_frames) in window_frames.iter().enumerate() {
                if i >= j {
                    continue;
                }
                // Two windows overlap iff they share any frame index.
                let any_shared = a_frames.iter().any(|af| b_frames.contains(af));
                assert!(
                    !any_shared,
                    "Mined windows {i} (frames {a_frames:?}) and {j} \
                     (frames {b_frames:?}) overlap — they share at least \
                     one frame index but must have separation ≥ \
                     {VERIFIER_WINDOW_SIZE}",
                );
            }
        }
    }

    #[test]
    fn test_mine_hard_negatives_constant_scores() {
        // Verify that mine_hard_negatives handles a classifier with
        // uniform (all-constant) scoring gracefully — every window
        // scores the same, so the first N in sorted order should be
        // selected (deterministic by position).
        use crate::wake_word_classifier::ClassifierWeights;
        let mut weights = ClassifierWeights::default();
        // Zero out all trainable weights to make output constant.
        weights.conv1_weight.fill(0.0);
        weights.conv1_bias.fill(0.0);
        weights.conv2_weight.fill(0.0);
        weights.conv2_bias.fill(0.0);
        weights.fc_weight.fill(0.0);
        // fc_bias = logit(0.9) so sigmoid(logit) ≈ 0.9
        let target = 0.9f32;
        weights.fc_bias[0] = (target / (1.0 - target)).ln();
        let classifier = WakeWordClassifier::new(weights);

        let mut rng = StdRng::seed_from_u64(400);
        let seq = make_seq_with_source(
            (0..20).map(|_| make_positive_frame(&mut rng)).collect(),
            crate::embedding_sequence::Source::Confusable,
        );

        let mined = mine_hard_negatives(
            &classifier,
            &[seq],
            2, // max_per_sequence
            VERIFIER_WINDOW_SIZE,
        );

        // Even with constant scores, should produce at most 2 windows.
        assert!(
            mined.sequences.len() <= 2,
            "Constant classifier should still produce at most 2 windows per sequence",
        );
        assert_eq!(
            mined.source_sequences_represented, 1,
            "Source sequence should be represented",
        );
    }
}
