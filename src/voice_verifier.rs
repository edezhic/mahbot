//! Small MLP verifier for wake word false-trigger suppression.
//!
//! Implements a lightweight second-stage classifier that runs AFTER the
//! Conv1D MLP classifier fires, as an additional AND gate. The verifier uses
//! a small 3-layer MLP (288 → 96 ReLU → 48 ReLU → 1 sigmoid) with optional
//! StandardScaler normalization (mahbot-861).
//!
//! The verifier operates on 3-frame stride-1 windows (288-dim = 3×96),
//! matching the classifier's windowing convention (mahbot-870).
//!
//! When not trained, the verifier acts as a no-op (all frames pass).
//!
//! # Architecture
//!
//! Training uses batch gradient descent on positive (enrollment) and negative
//! (synthetic or real) embedding examples with backpropagation. Inference is a
//! forward pass through the 3-layer MLP — ~3μs per frame.
//!
//! Backward compatibility: old logistic regression models (pre-mahbot-861) are
//! still supported for inference via `weights`/`bias` fields. New training
//! always produces an MLP.
//!
//! ## Training data
//!
//! - **Positive examples**: 3-frame stride-1 windows formed from enrollment
//!   utterance per-frame embeddings.
//! - **Negative examples**: Synthetic Gaussian noise (bootstrapping) or
//!   hard-negative embeddings collected from near-miss frames during detection.
//! - **Confusable negatives**: Pre-computed near-miss phrase embeddings (e.g.
//!   "hey map bot", "day mahbot") with 50-200× higher per-example weight during
//!   training so the verifier learns to reject confusable phrases.

use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use tracing::{info, warn};

/// Default decision threshold for the verifier (MLP decision boundary).
///
/// Calibrated empirically via threshold sweep (mahbot-881).  The sweep ran
/// HARD-tier benchmarks at 0.05 increments from 0.2 to 0.7 on the legacy TTS
/// negative path (`MAHBOT_BENCH_LEGACY_NEGATIVES=1`), with the top candidate
/// thresholds (0.25, 0.45, 0.50) replicated 2-4 times to estimate variance
/// from stochastic training.  The selected threshold maximises the mean
/// detection rate while satisfying all HARD-tier false-accept constraints
/// (confusable ≤1, noise ≤1, total ≤2), with MEDIUM and EASY tiers verified
/// at the calibrated value.
///
/// ## Sweep results
///
/// | Threshold | Runs | Detection rate (range) | HARD FA | HARD limits satisfied |
/// |-----------|------|----------------------|---------|----------------------|
/// | 0.20      | 1    | 53.8%                | 3       | ✗ (unrelated=2, total=3) |
/// | 0.25      | 2    | 46.2–61.5%           | 0       | ✓ |
/// | 0.30      | 1    | 53.8%                | 0       | ✓ |
/// | 0.35      | 1    | 53.8%                | 1       | ✗ (unrelated=1) |
/// | 0.40      | 1    | 53.8%                | 1       | ✓ |
/// | 0.45      | 3    | 38.5–69.2%           | 0       | ✓ |
/// | **0.50**  | 4    | **38.5–69.2%**       | 0       | ✓ |
/// | 0.55      | 1    | 38.5%                | 1       | ✓ |
/// | 0.60      | 1    | 46.2%                | 0       | ✓ |
/// | 0.65      | 1    | 46.2%                | 0       | ✓ |
/// | 0.70      | 1    | 46.2%                | 0       | ✓ |
///
/// **Selected: 0.50.**  Highest mean detection rate (57.7%) with zero false
/// accepts across all HARD runs.  MEDIUM tier: 30.8% (4/13), 0 FA (limits ✓).
/// EASY tier: 46.2% (6/13), 0 FA (limits ✓).  All false-accept limits
/// satisfied at all three tiers.
///
/// ⚠ **Note: 85% detection target not reached.**  The best threshold achieves
/// ~58% mean detection rate — the verifier training has high stochasticity
/// (20-30% variance between runs at the same threshold), and the underlying
/// classifier/verifier architecture may require further work (separate from
/// calibration) to hit the 85% target.
///
/// ⚠ **If changing this constant**, re-run the HARD-tier calibration sweep
/// first: `MAHBOT_VERIFIER_THRESHOLD=<val> cargo bench --bench voice_pipeline_e2e`.
/// Then verify MEDIUM and EASY tiers at the new value.
///
/// Previously at 0.4 (mahbot-853), 0.6 (mahbot-829), 0.5 (mahbot-797), and
/// 0.3 (mahbot-788).
pub(crate) const DEFAULT_VERIFIER_THRESHOLD: f32 = 0.50;

/// L2 regularization strength (lambda).
///
/// Reduced from 1.0 to 0.01 (mahbot-854) because the previous strong
/// regularization combined with extreme class imbalance (17:1 negatives-to-
/// positives) caused the model to learn constant near-zero outputs.  With
/// class-weighted loss now compensating for imbalance, weaker regularization
/// allows the model to develop discriminative weights.
pub(crate) const L2_LAMBDA: f32 = 0.01;

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
const MLP_HIDDEN_1: usize = 96;

/// MLP hidden layer 2 size (96 → 48).
const MLP_HIDDEN_2: usize = 48;

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
/// and rejecting the actual wake word.  Reduced to 50× as the documented
/// fallback, bringing confusable gradient to ~77% of total and restoring
/// positive class weight influence (mahbot-872).
pub(crate) const CONFUSABLE_UPWEIGHT: f32 = 50.0;

/// How much to upweight unrelated speech negative examples during MLP training.
///
/// Unrelated phrases (e.g. "what time is it", "good morning everyone") are
/// phonetically very different from the wake word but still represent real
/// non-wake-word speech that the verifier must reject.  10× gives them ~5×
/// more gradient contribution than ambient silence while still prioritising
/// confusable phrases as the primary negative signal.
pub(crate) const UNRELATED_UPWEIGHT: f32 = 10.0;

/// Embedding dimensionality (used by both verifier and voice pipeline).
pub(crate) const EMBEDDING_DIM: usize = 96;

/// Number of consecutive frames in a verifier window (matching classifier).
pub(crate) const VERIFIER_WINDOW_SIZE: usize = 3;

/// Input dimension for the verifier MLP: 3 × 96 = 288.
pub(crate) const VERIFIER_INPUT_DIM: usize = EMBEDDING_DIM * VERIFIER_WINDOW_SIZE;

/// Number of synthetic negative examples to generate for bootstrapping
/// when no real calibration data is available.
const SYNTHETIC_NEGATIVES_COUNT: usize = 100;

// ═══════════════════════════════════════════════════════════════════════════
// VoiceVerifier
// ═══════════════════════════════════════════════════════════════════════════

/// A lightweight MLP verifier for wake word false-trigger suppression
/// (mahbot-861).  Replaced the earlier logistic regression (mahbot-777).
///
/// Operates on 3-frame stride-1 windows (288-dim = 3×96), matching the
/// classifier's windowing convention (mahbot-870).
///
/// Architecture: 288 → 96 (ReLU) → 48 (ReLU) → 1 (sigmoid), with about 32,401
/// parameters.
///
/// Computes `MLP(L2_norm(scaler(x)))` for a given 288-dim window, where the
/// scaler is a StandardScaler fitted during training.  If the score is below
/// `threshold`, the wake word detection is suppressed.
///
/// Backward compatibility: old serialized models with `weights`+`bias` (linear)
/// that match the 288-dim input dimension are still supported at inference time.
/// Legacy 96-dim models are correctly rejected by `is_trained()` since they cannot
/// process 288-dim windows (mahbot-870).
///
/// When `trained` is `false`, the verifier is a no-op (all frames pass with
/// score 1.0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceVerifier {
    /// Legacy logistic regression weights (backward compat, mahbot-777).
    #[serde(default)]
    pub weights: Vec<f32>,
    /// Legacy logistic regression bias (backward compat).
    #[serde(default)]
    pub bias: f32,

    // ── MLP weights (mahbot-861) ─────────────────────────────────────────
    // Architecture: 288 → 96 (ReLU) → 48 (ReLU) → 1 (sigmoid)
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

    /// Predict the probability that the given 288-dim window is a genuine
    /// wake word.
    ///
    /// Operates on 3-frame stride-1 windows (288-dim = 3×96), matching the
    /// classifier's windowing convention (mahbot-870).
    ///
    /// Returns a score in `[0.0, 1.0]`. When untrained, always returns `1.0`
    /// (no-op — all frames pass).
    ///
    /// Uses the MLP forward pass when MLP parameters are available (new format,
    /// mahbot-861), falls back to the legacy logistic regression (linear) for
    /// backward compatibility with old serialized models (mahbot-777).
    #[must_use]
    pub fn predict(&self, embedding: &[f32]) -> f32 {
        if !self.is_trained() {
            return 1.0;
        }

        // Validate embedding dimension matches expected input dimension (288).
        if embedding.len() != VERIFIER_INPUT_DIM {
            warn!(
                "Verifier embedding dimension mismatch: got {}, expected {}; falling back to no-op",
                embedding.len(),
                VERIFIER_INPUT_DIM,
            );
            return 1.0;
        }

        // ═══════════════════════════════════════════════════════════════
        // Pipeline: raw 288-dim window → L2-norm → scaler → MLP/linear
        // ═══════════════════════════════════════════════════════════════
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
        // MLP dimension validation prevents index-out-of-bounds in
        // mlp_forward() if a corrupted serialized model has wrong-weight tensors.
        //
        // Note: Legacy 96-dim linear models are not supported through this
        // fallback — is_trained() correctly rejects them because their weights
        // dimension (96) does not match VERIFIER_INPUT_DIM (288).  Only
        // 288-dim linear models (which no production training produces) are
        // reachable here (mahbot-874).
        //
        // MLP path: input is L2-norm → scaler (matching training pipeline).
        // Linear path: legacy models predate both L2-norm and scaler, so use
        // raw embedding directly.
        if self.has_valid_mlp_params() {
            mlp_forward(
                &x, &self.w1, &self.b1, &self.w2, &self.b2, &self.w3, self.b3,
            )
        } else if !self.weights.is_empty() {
            // Legacy linear combination: z = w·x + b.
            let z: f32 = embedding
                .iter()
                .zip(self.weights.iter())
                .map(|(x, w)| x * w)
                .sum::<f32>()
                + self.bias;
            // Sigmoid activation
            sigmoid(z)
        } else {
            // is_trained() guarantees either MLP or linear weights exist, so
            // this branch is structurally unreachable.
            unreachable!(
                "predict() called on trained verifier with neither MLP nor linear weights"
            );
        }
    }

    /// Train a new verifier from positive and negative 96-dim embedding
    /// examples using a small MLP with L2 regularization (mahbot-861).
    ///
    /// Internally forms 3-frame stride-1 windows (288-dim) from the per-frame
    /// embeddings, matching the classifier's windowing convention (mahbot-870).
    /// Each window is L2-normalized before MLP training.
    ///
    /// Architecture: 288 → 96 (ReLU) → 48 (ReLU) → 1 (sigmoid), ~32,401 params.
    ///
    /// # Arguments
    ///
    /// * `positive_embeddings` — Per-frame embeddings from enrollment
    ///   utterances (label = 1). Each element is a single 96-dim vector.
    /// * `negative_embeddings` — Embeddings from non-wake-word audio
    ///   (label = 0). Each element is a single 96-dim vector.
    /// * `per_negative_weights` — Optional per-example weights for *negative*
    ///   samples only (used to upweight confusable near-miss phrases).  When
    ///   `Some(weights)`, `weights.len()` must equal `negative_embeddings.len()`.
    ///   Positives are weighted by the automatic `n_neg / n_pos` class_weight.
    /// * `threshold` — Decision threshold (defaults to
    ///   [`DEFAULT_VERIFIER_THRESHOLD`] in production).
    /// * `l2_lambda` — L2 regularisation strength.
    /// * `learning_rate` — Gradient descent learning rate.
    /// * `max_iter` — Maximum gradient descent iterations.
    ///
    /// Returns a trained `VoiceVerifier`, or an untrained verifier if either
    /// input list is empty.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn train(
        positive_embeddings: &[Vec<f32>],
        negative_embeddings: &[Vec<f32>],
        per_negative_weights: Option<&[f32]>,
        threshold: f32,
        l2_lambda: f32,
        learning_rate: f32,
        max_iter: usize,
    ) -> Self {
        if positive_embeddings.is_empty() || negative_embeddings.is_empty() {
            warn!(
                "Cannot train verifier: need both positive ({}) and negative ({}) examples",
                positive_embeddings.len(),
                negative_embeddings.len(),
            );
            return Self::untrained();
        }

        // Validate per-negative weights length and fall through to None on
        // mismatch so the caller gets an unambiguous error (silently wrong
        // weights would silently produce a wrong model, mahbot-861).
        let weights_to_use = match per_negative_weights {
            Some(w) if w.len() == negative_embeddings.len() => Some(w),
            Some(w) => {
                warn!(
                    "per_negative_weights length ({}) does not match negative_embeddings length ({}); \
                     falling back to uniform (1.0) negative weights",
                    w.len(),
                    negative_embeddings.len(),
                );
                None
            }
            None => None,
        };

        let dim = positive_embeddings[0].len();
        if dim == 0 {
            return Self::untrained();
        }

        let n_pos = positive_embeddings.len();
        let n_neg = negative_embeddings.len();

        // Compute class weight to compensate for imbalance: amplify
        // positive-sample gradients so the model is penalised equally for
        // misclassifying positives and negatives.
        let class_weight = {
            #[allow(clippy::cast_precision_loss)]
            let n_pos_f = n_pos as f32;
            #[allow(clippy::cast_precision_loss)]
            let n_neg_f = n_neg as f32;
            if n_pos_f > 0.0 {
                n_neg_f / n_pos_f
            } else {
                1.0
            }
        };

        // Combine positive (label = 1.0) and negative (label = 0.0) examples
        let mut features: Vec<Vec<f32>> = Vec::with_capacity(n_pos + n_neg);
        let mut labels: Vec<f32> = Vec::with_capacity(n_pos + n_neg);

        for emb in positive_embeddings {
            features.push(emb.clone());
            labels.push(1.0);
        }
        for emb in negative_embeddings {
            features.push(emb.clone());
            labels.push(0.0);
        }

        // ── Form 3-frame stride-1 windows ──
        // Convert the flat list of 96-dim per-frame embeddings into 288-dim
        // stride-1 windows (each window is 3 consecutive frames concatenated).
        // This matches the classifier's windowing convention and gives the
        // verifier temporal context to distinguish confusable phrases.
        //
        // If input features are already at VERIFIER_INPUT_DIM (e.g., caller
        // has pre-windowed them), skip concatenation and just L2-normalize.
        let (windowed_features, windowed_labels, windowed_sample_weights, n_pos_w, n_neg_w) =
            if !features.is_empty() && features[0].len() == EMBEDDING_DIM {
                // Per-frame input: form stride-1 windows.
                let windows = form_stride1_windows(&features);
                if windows.is_empty() {
                    warn!(
                        "Cannot form windows: need at least {} per-frame embeddings, got {}",
                        VERIFIER_WINDOW_SIZE,
                        features.len(),
                    );
                    return Self::untrained();
                }

                // Assign labels based on the center frame (index 1) of each window.
                // For stride-1 windows, window k has center at features[k + 1].
                // This conservatively labels boundary-crossing windows: at most
                // 2 windows at the pos/neg boundary get a potentially mixed label,
                // which is negligible for training.
                let w_labels: Vec<f32> = (0..windows.len()).map(|i| labels[i + 1]).collect();

                // Build per-sample weights for windows (use center frame weight).
                let n_pos_frames = positive_embeddings.len();
                let sample_weights =
                    build_sample_weights(n_pos_frames, n_neg, class_weight, weights_to_use);
                let w_weights: Vec<f32> =
                    (0..windows.len()).map(|i| sample_weights[i + 1]).collect();

                // Count positive/negative windows (center-frame label).
                let n_pw = w_labels.iter().filter(|&&l| l > 0.5).count();
                let n_neg_w = w_labels.len() - n_pw;

                (windows, w_labels, w_weights, n_pw, n_neg_w)
            } else {
                // Already windowed (e.g., test data at VERIFIER_INPUT_DIM).
                // Just L2-normalize each feature vector.
                let normalized: Vec<Vec<f32>> = features
                    .iter()
                    .map(|f| {
                        let norm = f.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
                        f.iter().map(|v| v / norm).collect()
                    })
                    .collect();

                let sample_weights =
                    build_sample_weights(n_pos, n_neg, class_weight, weights_to_use);

                (normalized, labels, sample_weights, n_pos, n_neg)
            };

        // 1. Compute StandardScaler (per-dimension mean and std) on windowed features.
        let (scaler_mean, scaler_std) = compute_standard_scaler(&windowed_features);

        // 2. Apply scaling to all training features.
        let scaled_features: Vec<Vec<f32>> = windowed_features
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

        // 3. Train MLP on scaled windowed features.
        let MlpWeights {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
        } = train_mlp(
            &scaled_features,
            &windowed_labels,
            VERIFIER_INPUT_DIM,
            &windowed_sample_weights,
            l2_lambda,
            learning_rate,
            max_iter,
        );

        // 4. Build trained verifier with MLP parameters.
        let verifier = Self {
            weights: Vec::new(), // MLP format — leave legacy fields empty
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
            trained: true,
        };

        // 5. Training diagnostics: compute scores on windowed training examples
        //    so we can verify the model actually discriminates (mahbot-854).
        {
            // Use windowed features for diagnostics since predict() now expects
            // 288-dim windows.
            let mut pos_scores = Vec::with_capacity(n_pos_w);
            let mut neg_scores = Vec::with_capacity(n_neg_w);
            for (emb, &lbl) in windowed_features.iter().zip(windowed_labels.iter()) {
                let score = verifier.predict(emb);
                if lbl > 0.5 {
                    pos_scores.push(score);
                } else {
                    neg_scores.push(score);
                }
            }

            #[allow(clippy::cast_precision_loss)]
            let pos_mean = if pos_scores.is_empty() {
                0.0
            } else {
                pos_scores.iter().sum::<f32>() / n_pos_w as f32
            };
            #[allow(clippy::cast_precision_loss)]
            let neg_mean = if neg_scores.is_empty() {
                0.0
            } else {
                neg_scores.iter().sum::<f32>() / n_neg_w as f32
            };
            let pos_min = pos_scores.iter().copied().fold(f32::INFINITY, f32::min);
            let pos_max = pos_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let neg_min = neg_scores.iter().copied().fold(f32::INFINITY, f32::min);
            let neg_max = neg_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            info!(
                "Verifier training diagnostics: {n_pos_w} pos + {n_neg_w} neg windows, \
                 class_weight={class_weight:.2}, L2={l2_lambda}, LR={learning_rate}, \
                 max_iter={max_iter} | \
                 pos scores: mean={pos_mean:.4} [{pos_min:.4}, {pos_max:.4}] | \
                 neg scores: mean={neg_mean:.4} [{neg_min:.4}, {neg_max:.4}]",
            );
        }

        verifier
    }

    /// Convenience: train a verifier using the given positive embeddings and
    /// automatically generated synthetic negative examples (distribution-
    /// matched via [`generate_synthetic_negatives_from_positives`] instead of
    /// pure N(0,1) Gaussian noise).
    ///
    /// Uses default MLP_MAX_ITER for training since synthetic negatives don't
    /// need confusable upweighting.
    #[must_use]
    pub fn train_with_synthetic_negatives(
        positive_embeddings: &[Vec<f32>],
        threshold: f32,
    ) -> Self {
        let negatives = generate_synthetic_negatives_from_positives(
            SYNTHETIC_NEGATIVES_COUNT,
            positive_embeddings,
            1.5, // noise_scale — matched to benchmark default
        );
        Self::train(
            positive_embeddings,
            &negatives,
            None, // no per-negative weights for synthetic negatives
            threshold,
            L2_LAMBDA,
            LEARNING_RATE,
            MLP_MAX_ITER,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Window helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Build the per-sample weights array for MLP training.
///
/// - Positives: all get `class_weight` (auto imbalance compensation).
/// - Negatives: use `per_negative_weights[i]` if provided, else 1.0.
fn build_sample_weights(
    n_pos: usize,
    n_neg: usize,
    class_weight: f32,
    per_negative_weights: Option<&[f32]>,
) -> Vec<f32> {
    let mut w = Vec::with_capacity(n_pos + n_neg);
    for _ in 0..n_pos {
        w.push(class_weight);
    }
    match per_negative_weights {
        Some(pnw) => w.extend_from_slice(pnw),
        None => w.extend(std::iter::repeat_n(1.0, n_neg)),
    }
    w
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

/// Form stride-1 windows from a flat list of 96-dim embeddings.
///
/// Each window is 3 consecutive embeddings concatenated into a 288-dim vector,
/// then L2-normalized (matching classifier convention).  Consecutive windows
/// overlap by 2 embeddings (stride 1).
///
/// Returns empty vec if fewer than 3 embeddings are available.
fn form_stride1_windows(embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if embeddings.len() < VERIFIER_WINDOW_SIZE {
        return Vec::new();
    }
    let mut windows = Vec::with_capacity(embeddings.len() - VERIFIER_WINDOW_SIZE + 1);
    for i in 0..=embeddings.len() - VERIFIER_WINDOW_SIZE {
        let mut window = vec![0.0f32; VERIFIER_INPUT_DIM];
        fill_verifier_window(embeddings, i, &mut window);
        // L2-normalize the window (matching classifier convention, mahbot-870).
        let norm = window.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
        for v in &mut window {
            *v /= norm;
        }
        windows.push(window);
    }
    windows
}

// ═══════════════════════════════════════════════════════════════════════════
// Math helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Standard sigmoid function: `1 / (1 + e^{-x})`.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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

/// Forward pass through the 3-layer MLP (288 → 96 ReLU → 48 ReLU → 1 sigmoid).
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

    // Layer 1: h1 = ReLU(W1^T · x + b1)
    // w1 is stored row-major: w1[i * h1_size + j] = weight from x[i] to h1[j]
    // So h1[j] = sum_i x[i] * w1[i * h1_size + j] + b1[j]
    let mut h1 = vec![0.0; h1_size];
    for j in 0..h1_size {
        let mut s = b1[j];
        for i in 0..input_dim {
            s += x[i] * w1[i * h1_size + j];
        }
        h1[j] = if s > 0.0 { s } else { 0.0 }; // ReLU
    }

    // Layer 2: h2 = ReLU(W2^T · h1 + b2)
    let mut h2 = vec![0.0; h2_size];
    for k in 0..h2_size {
        let mut s = b2[k];
        for j in 0..h1_size {
            s += h1[j] * w2[j * h2_size + k];
        }
        h2[k] = if s > 0.0 { s } else { 0.0 }; // ReLU
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

/// Train a small MLP (288 → 96 → 48 → 1) using batch gradient descent with
/// L2 regularization and per-sample weighting (mahbot-861).
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

    for w in &mut w1 {
        *w = rand::random::<f32>() * 2.0 * w1_bound - w1_bound;
    }
    for w in &mut w2 {
        *w = rand::random::<f32>() * 2.0 * w2_bound - w2_bound;
    }
    for w in &mut w3 {
        *w = rand::random::<f32>() * 2.0 * w3_bound - w3_bound;
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

            // h1 = ReLU(W1^T · x + b1)
            for j in 0..h1_size {
                let mut s = b1[j];
                for k in 0..dim {
                    s += x[k] * w1[k * h1_size + j];
                }
                h1_pre[j] = s;
                h1[j] = if s > 0.0 { s } else { 0.0 };
            }

            // h2 = ReLU(W2^T · h1 + b2)
            for k in 0..h2_size {
                let mut s = b2[k];
                for j in 0..h1_size {
                    s += h1[j] * w2[j * h2_size + k];
                }
                h2_pre[k] = s;
                h2[k] = if s > 0.0 { s } else { 0.0 };
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
                // ReLU derivative: d(h2)/d(pre) = 1 if pre > 0 else 0
                if h2_pre[k] <= 0.0 {
                    dh2[k] = 0.0;
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
                // ReLU derivative
                if h1_pre[j] <= 0.0 {
                    dh1[j] = 0.0;
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
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub(crate) fn generate_synthetic_negatives_from_positives(
    count: usize,
    positives: &[Vec<f32>],
    noise_scale: f32,
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

    (0..count)
        .map(|_| {
            // Pick a random positive as the base (adds diversity).
            let base = &positives[rand::rng().random_range(0..positives.len())];
            let mut emb: Vec<f32> = base
                .iter()
                .zip(std.iter())
                .map(|(&b, &s)| {
                    // Box-Muller N(0,1)
                    let z = loop {
                        let u1: f32 = rand::random();
                        let u2: f32 = rand::random();
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

        let verifier = VoiceVerifier::train(
            &positives,
            &negatives,
            None,                       // no per-negative weights
            DEFAULT_VERIFIER_THRESHOLD, // threshold
            0.001,                      // weak L2 (clean synthetic data)
            0.1,                        // learning rate
            500,                        // max iter
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

        let verifier = VoiceVerifier::train(
            &positives,
            &negatives,
            None,                       // no per-negative weights
            DEFAULT_VERIFIER_THRESHOLD, // mahbot-853: lowered from 0.6 for streaming inference.
            L2_LAMBDA,                  // L2 regularization (mahbot-854: 0.01)
            LEARNING_RATE,              // learning rate (mahbot-878: 0.001, Adam)
            MAX_ITER,                   // max iterations (mahbot-854: 5000)
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

        let verifier =
            VoiceVerifier::train_with_synthetic_negatives(&positives, DEFAULT_VERIFIER_THRESHOLD);

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

        let verifier = VoiceVerifier::train(
            &positives,
            &negatives,
            None, // no per-negative weights
            DEFAULT_VERIFIER_THRESHOLD,
            0.001,
            0.1,
            500,
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
        let negs = generate_synthetic_negatives_from_positives(10, &positives, 1.5);
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
        let negs = generate_synthetic_negatives_from_positives(0, &positives, 1.5);
        assert!(negs.is_empty());
    }

    #[test]
    fn test_generate_synthetic_negatives_from_positives_empty_positives() {
        let positives: Vec<Vec<f32>> = vec![];
        let negs = generate_synthetic_negatives_from_positives(10, &positives, 1.5);
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
        let negs = generate_synthetic_negatives_from_positives(200, &positives, 1.0);
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
        let verifier =
            VoiceVerifier::train_with_synthetic_negatives(&positives, DEFAULT_VERIFIER_THRESHOLD);
        assert!(verifier.is_trained());
        assert_eq!(
            verifier.threshold, DEFAULT_VERIFIER_THRESHOLD,
            "threshold must match DEFAULT_VERIFIER_THRESHOLD",
        );
        assert_eq!(verifier.w1.len(), VERIFIER_INPUT_DIM * MLP_HIDDEN_1);
        assert_eq!(verifier.b1.len(), MLP_HIDDEN_1);
        assert!(!verifier.scaler_mean.is_empty());
        assert!(!verifier.scaler_std.is_empty());

        // All MLP weights must be finite — NaN/inf indicates gradient divergence
        // from unstable hyperparameters.
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

        let verifier = VoiceVerifier::train(
            &positives,
            &negatives,
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            0.001,
            0.1,
            500,
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
        let verifier = VoiceVerifier::train(
            &[],
            &[vec![0.0; VERIFIER_INPUT_DIM]],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            0.001,
            0.1,
            100,
        );
        assert!(!verifier.is_trained());
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
}
