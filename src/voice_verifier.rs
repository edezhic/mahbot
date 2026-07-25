//! Small MLP verifier for wake word false-trigger suppression.
//!
//! Implements a lightweight second-stage classifier that runs AFTER the
//! Conv1D MLP classifier fires, as an additional AND gate. The verifier uses
//! a small 3-layer MLP (96 → 32 ReLU → 16 ReLU → 1 sigmoid) with optional
//! StandardScaler normalization (mahbot-861).
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
//! - **Positive examples**: Mean-pooled 96-dim embeddings from each enrollment
//!   utterance (10 per enrollment).
//! - **Negative examples**: Synthetic Gaussian noise (bootstrapping) or
//!   hard-negative embeddings collected from near-miss frames during detection.
//! - **Confusable negatives**: Pre-computed near-miss phrase embeddings (e.g.
//!   "hey map bot", "day mahbot") with 50-200× higher per-example weight during
//!   training so the verifier learns to reject confusable phrases.

use rand::RngExt;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Default decision threshold for the verifier (standard logistic regression
/// decision boundary).
///
/// Set to 0.4 (mahbot-853) — streaming-inference embeddings produce different
/// feature statistics than training extraction (VAD-gated chunks, alignment
/// padding artifacts), so the original 0.6 threshold rejected all streaming
/// detections even for the enrolled speaker.  The classifier already provides
/// strong discrimination (pos_scores_mean ~0.93 vs neg_scores_mean ~0.04), so
/// the verifier can be more permissive.  Previously at 0.6 (mahbot-829), 0.5
/// (mahbot-797), and 0.3 (mahbot-788).
pub(crate) const DEFAULT_VERIFIER_THRESHOLD: f32 = 0.4;

/// L2 regularization strength (lambda).
///
/// Reduced from 1.0 to 0.01 (mahbot-854) because the previous strong
/// regularization combined with extreme class imbalance (17:1 negatives-to-
/// positives) caused the model to learn constant near-zero outputs.  With
/// class-weighted loss now compensating for imbalance, weaker regularization
/// allows the model to develop discriminative weights.
pub(crate) const L2_LAMBDA: f32 = 0.01;

/// Learning rate for gradient descent.
pub(crate) const LEARNING_RATE: f32 = 0.01;

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

/// MLP hidden layer 1 size (96 → 32).
const MLP_HIDDEN_1: usize = 32;

/// MLP hidden layer 2 size (32 → 16).
const MLP_HIDDEN_2: usize = 16;

/// Maximum iterations for MLP verifier training.
///
/// The MLP converges faster per-iteration than logistic regression because
/// the non-linear hidden layers provide richer gradient signal.  Set to 2000
/// as a balance between convergence quality and training latency (<1s with
/// ~2655 training examples and ~3649 parameters).
pub(crate) const MLP_MAX_ITER: usize = 2000;

/// How much to upweight confusable negative examples during MLP training.
///
/// Confusable phrases (e.g. "hey map bot", "day mahbot") are acoustically
/// very similar to the wake word.  Without this upweighting, their gradient
/// signal is drowned out by thousands of ambient negatives.  100× gives them
/// ~50% of total negative gradient contribution despite being <5% of negatives.
pub(crate) const CONFUSABLE_UPWEIGHT: f32 = 100.0;

/// Embedding dimensionality (used by both verifier and voice pipeline).
pub(crate) const EMBEDDING_DIM: usize = 96;

/// Number of synthetic negative examples to generate for bootstrapping
/// when no real calibration data is available.
const SYNTHETIC_NEGATIVES_COUNT: usize = 100;

// ═══════════════════════════════════════════════════════════════════════════
// VoiceVerifier
// ═══════════════════════════════════════════════════════════════════════════

/// A lightweight MLP verifier for wake word false-trigger suppression
/// (mahbot-861).  Replaced the earlier logistic regression (mahbot-777).
///
/// Architecture: 96 → 32 (ReLU) → 16 (ReLU) → 1 (sigmoid), with about 3,649
/// parameters.
///
/// Computes `MLP(scaler(x))` for a given 96-dim embedding, where the scaler
/// is a StandardScaler fitted during training.  If the score is below
/// `threshold`, the wake word detection is suppressed.
///
/// Backward compatibility: old serialized models with `weights`+`bias` (linear)
/// are still supported at inference time.
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
    // Architecture: 96 → 32 (ReLU) → 16 (ReLU) → 1 (sigmoid)
    //
    // Storage convention (row-major):
    //   w1[i * MLP_HIDDEN_1 + j] = weight from input[i] to hidden1[j]
    //   w2[j * MLP_HIDDEN_2 + k] = weight from hidden1[j] to hidden2[k]
    //   w3[k]                     = weight from hidden2[k] to output
    /// Layer 1 weights: 96 × 32 (row-major).
    #[serde(default)]
    pub w1: Vec<f32>,
    /// Layer 1 biases: 32.
    #[serde(default)]
    pub b1: Vec<f32>,
    /// Layer 2 weights: 32 × 16 (row-major).
    #[serde(default)]
    pub w2: Vec<f32>,
    /// Layer 2 biases: 16.
    #[serde(default)]
    pub b2: Vec<f32>,
    /// Layer 3 weights: 16 × 1.
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
    /// scaler dimensions match the input dimension (96).  MLP weight/bias
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
        if !has_valid_mlp && has_linear && self.weights.len() != EMBEDDING_DIM {
            return false;
        }
        // If either scaler is non-empty, both must be present and match the
        // 96-dim input.
        let input_dim = EMBEDDING_DIM;
        if (!self.scaler_mean.is_empty() || !self.scaler_std.is_empty())
            && (self.scaler_mean.len() != input_dim || self.scaler_std.len() != input_dim)
        {
            return false;
        }
        true
    }

    /// Returns `true` if MLP weight/bias tensor dimensions match the
    /// 96→32→16→1 architecture and `b3` is finite (no NaN/Inf which would
    /// produce `sigmoid(NaN)=0.5` silently).
    ///
    /// Only `b3` (the scalar output bias) is checked for finiteness: it's a
    /// single float, so the check is free, and a NaN there would corrupt every
    /// prediction (sigmoid output pinned to 0.5) — the most dangerous single
    /// failure point from serialization corruption.  Weight tensors are not
    /// individually validated for NaN/Inf because element-wise checks on ~3600
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
        if self.w1.len() != EMBEDDING_DIM * MLP_HIDDEN_1
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

    /// Production decision threshold.
    ///
    /// Returns the value of [`DEFAULT_VERIFIER_THRESHOLD`] — the threshold
    /// used by all production enrollment paths.  Tests should reference this
    /// method instead of hardcoding a literal threshold value.
    #[must_use]
    pub fn default_threshold() -> f32 {
        DEFAULT_VERIFIER_THRESHOLD
    }

    /// Predict the probability that the given 96-dim embedding is a genuine
    /// wake word.
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

        // Validate embedding dimension matches expected input dimension (96).
        if embedding.len() != EMBEDDING_DIM {
            warn!(
                "Verifier embedding dimension mismatch: got {}, expected {}; falling back to no-op",
                embedding.len(),
                EMBEDDING_DIM,
            );
            return 1.0;
        }

        // Apply StandardScaler normalisation if available (both mean and std
        // must be populated from training).
        let x: Vec<f32> = if !self.scaler_mean.is_empty() && !self.scaler_std.is_empty() {
            embedding
                .iter()
                .zip(self.scaler_mean.iter())
                .zip(self.scaler_std.iter())
                .map(
                    |((&val, &mean), &std)| {
                        if std > 0.0 { (val - mean) / std } else { val }
                    },
                )
                .collect()
        } else {
            embedding.to_vec()
        };

        // Use MLP if available (new format, mahbot-861), fall back to legacy
        // linear model for backward compatibility.  MLP dimension validation
        // prevents index-out-of-bounds in mlp_forward() if a corrupted
        // serialized model has wrong-weight tensors.
        if self.has_valid_mlp_params() {
            mlp_forward(
                &x, &self.w1, &self.b1, &self.w2, &self.b2, &self.w3, self.b3,
            )
        } else if !self.weights.is_empty() {
            // Legacy linear combination: z = w·x + b
            let z: f32 = x
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
    /// Architecture: 96 → 32 (ReLU) → 16 (ReLU) → 1 (sigmoid), ~3,649 params.
    ///
    /// # Arguments
    ///
    /// * `positive_embeddings` — Mean-pooled embeddings from enrollment
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

        // 1. Compute StandardScaler (per-dimension mean and std)
        let (scaler_mean, scaler_std) = compute_standard_scaler(&features);

        // 2. Apply scaling to all training features
        let scaled_features: Vec<Vec<f32>> = features
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

        // 3. Compute class weight to compensate for imbalance: amplify
        //    positive-sample gradients so the model is penalised equally for
        //    misclassifying positives and negatives.
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

        // 4. Build per-sample weights array.
        //    - Positives: all get `class_weight` (auto imbalance compensation).
        //    - Negatives: use `weights_to_use[i]` if provided, else 1.0.
        let sample_weights: Vec<f32> = {
            let mut w = Vec::with_capacity(n_pos + n_neg);
            for _ in 0..n_pos {
                w.push(class_weight);
            }
            match weights_to_use {
                Some(pnw) => w.extend_from_slice(pnw),
                None => w.extend(std::iter::repeat_n(1.0, n_neg)),
            }
            w
        };

        // 5. Train MLP on scaled features
        let MlpWeights {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
        } = train_mlp(
            &scaled_features,
            &labels,
            dim,
            &sample_weights,
            l2_lambda,
            learning_rate,
            max_iter,
        );

        // 6. Build trained verifier with MLP parameters
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

        // 7. Training diagnostics: compute scores on training examples so we
        //    can verify the model actually discriminates (mahbot-854).
        {
            let mut pos_scores = Vec::with_capacity(n_pos);
            let mut neg_scores = Vec::with_capacity(n_neg);
            for emb in positive_embeddings {
                pos_scores.push(verifier.predict(emb));
            }
            for emb in negative_embeddings {
                neg_scores.push(verifier.predict(emb));
            }

            #[allow(clippy::cast_precision_loss)]
            let pos_mean = pos_scores.iter().sum::<f32>() / n_pos as f32;
            #[allow(clippy::cast_precision_loss)]
            let neg_mean = neg_scores.iter().sum::<f32>() / n_neg as f32;
            let pos_min = pos_scores.iter().copied().fold(f32::INFINITY, f32::min);
            let pos_max = pos_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let neg_min = neg_scores.iter().copied().fold(f32::INFINITY, f32::min);
            let neg_max = neg_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            info!(
                "Verifier training diagnostics: {n_pos} pos + {n_neg} neg examples, \
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

/// Forward pass through the 3-layer MLP (96 → 32 ReLU → 16 ReLU → 1 sigmoid).
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

/// Trained MLP weights and biases for the 96→32→16→1 voice verifier
/// (mahbot-861).  Returned by [`train_mlp`] to provide compile-time
/// argument-order safety when assigning fields to the [`VoiceVerifier`].
struct MlpWeights {
    /// 96 × 32, row-major: `w1[i * 32 + j]` = weight from input `i` to h1 `j`.
    w1: Vec<f32>,
    /// 32 bias terms for h1.
    b1: Vec<f32>,
    /// 32 × 16, row-major: `w2[j * 16 + k]` = weight from h1 `j` to h2 `k`.
    w2: Vec<f32>,
    /// 16 bias terms for h2.
    b2: Vec<f32>,
    /// 16 × 1, flat: `w3[k]` = weight from h2 `k` to output.
    w3: Vec<f32>,
    /// Scalar output bias.
    b3: f32,
}

/// Train a small MLP (96 → 32 → 16 → 1) using batch gradient descent with
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
    clippy::too_many_lines
)]
fn train_mlp(
    features: &[Vec<f32>],  // scaled (n × dim)
    labels: &[f32],         // 0.0 or 1.0
    dim: usize,             // input dimension (96)
    sample_weights: &[f32], // per-sample weight (n)
    l2_lambda: f32,
    learning_rate: f32,
    max_iter: usize,
) -> MlpWeights {
    let n = features.len();
    let h1_size = MLP_HIDDEN_1; // 32
    let h2_size = MLP_HIDDEN_2; // 16

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
    // w1: 96 × 32, stored row-major: w1[i * h1_size + j]
    let mut w1 = vec![0.0; dim * h1_size];
    let mut b1 = vec![0.0; h1_size];
    // w2: 32 × 16, stored row-major: w2[j * h2_size + k]
    let mut w2 = vec![0.0; h1_size * h2_size];
    let mut b2 = vec![0.0; h2_size];
    // w3: 16 × 1
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
    // re-allocation churn: ~29 MB over 2000 iterations with ~3649 params).
    let mut dw1 = vec![0.0; dim * h1_size];
    let mut db1 = vec![0.0; h1_size];
    let mut dw2 = vec![0.0; h1_size * h2_size];
    let mut db2 = vec![0.0; h2_size];
    let mut dw3 = vec![0.0; h2_size];

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

        // ── Gradient descent update ──
        for j in 0..w1.len() {
            w1[j] -= learning_rate * dw1[j];
        }
        for j in 0..b1.len() {
            b1[j] -= learning_rate * db1[j];
        }
        for j in 0..w2.len() {
            w2[j] -= learning_rate * dw2[j];
        }
        for j in 0..b2.len() {
            b2[j] -= learning_rate * db2[j];
        }
        for j in 0..w3.len() {
            w3[j] -= learning_rate * dw3[j];
        }
        b3 -= learning_rate * db3;
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
    /// normalization step.
    ///
    /// The caller is responsible for applying the scaler (if any) before
    /// calling this function.  This is used by consistency tests that compare
    /// the verifier output against `mlp_forward()` directly without
    /// duplicating the scaling logic.
    ///
    /// Routes to the MLP if parameters are valid, falling back to the legacy
    /// linear model.  Does **not** check `is_trained()` — callers should
    /// ensure the verifier is trained before calling this function.
    fn predict_scaled(verifier: &VoiceVerifier, scaled: &[f32]) -> f32 {
        if verifier.has_valid_mlp_params() {
            mlp_forward(
                scaled,
                &verifier.w1,
                &verifier.b1,
                &verifier.w2,
                &verifier.b2,
                &verifier.w3,
                verifier.b3,
            )
        } else {
            // Legacy linear combination.
            let z: f32 = scaled
                .iter()
                .zip(verifier.weights.iter())
                .map(|(x, w)| x * w)
                .sum::<f32>()
                + verifier.bias;
            sigmoid(z)
        }
    }

    /// Generate a synthetic 96-dim "positive" embedding with values clustered
    /// around +0.5 (simulating a wake-word embedding).
    fn make_positive_embedding(rng: &mut impl Rng) -> Vec<f32> {
        (0..EMBEDDING_DIM)
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

    /// Generate a synthetic 96-dim "negative" embedding with values clustered
    /// around -0.5 (simulating a non-wake-word embedding).
    fn make_negative_embedding(rng: &mut impl Rng) -> Vec<f32> {
        (0..EMBEDDING_DIM)
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

    /// Generate a synthetic 96-dim "non-wake-word" embedding with values
    /// distributed near 0 (simulating real non-wake-word speech or ambient
    /// audio that survives Conv1D MLP matching).  Unlike the old opposite-direction
    /// negatives (N(-0.5, 0.3)), these sit in the same general region as
    /// wake word embeddings but lack the consistent structure that the
    /// verifier must learn to discriminate (mahbot-797).
    fn make_non_wake_speech_embedding(rng: &mut impl Rng) -> Vec<f32> {
        (0..EMBEDDING_DIM)
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

    #[test]
    fn test_verifier_accepts_positive() {
        // Train on known positive and negative synthetic embeddings.
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
        let held_out = make_positive_embedding(&mut rng);
        let score = verifier.predict(&held_out);
        assert!(
            score >= 0.5,
            "Verifier should accept positive embedding (score >= 0.5), got score={score:.4}",
        );
    }

    #[test]
    fn test_verifier_rejects_negative() {
        let mut rng = StdRng::seed_from_u64(42);
        let positives: Vec<Vec<f32>> = (0..20).map(|_| make_positive_embedding(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..30).map(|_| make_negative_embedding(&mut rng)).collect();

        let verifier = VoiceVerifier::train(
            &positives,
            &negatives,
            None, // no per-negative weights
            DEFAULT_VERIFIER_THRESHOLD,
            0.001,
            0.1,
            500,
        );

        assert!(verifier.is_trained());

        // Verify a held-out negative is rejected.
        let held_out = make_negative_embedding(&mut rng);
        let score = verifier.predict(&held_out);
        assert!(
            score < 0.5,
            "Verifier should reject negative embedding (score < 0.5), got score={score:.4}",
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
            LEARNING_RATE,              // learning rate (mahbot-854: 0.01)
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
        let embedding = vec![0.5; EMBEDDING_DIM];
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
            weights: vec![0.5; 96],
            bias: 0.0,
            w1: Vec::new(),
            b1: Vec::new(),
            w2: Vec::new(),
            b2: Vec::new(),
            w3: Vec::new(),
            b3: 0.0,
            scaler_mean: vec![0.1; 32], // wrong dimension (32 ≠ 96)
            scaler_std: vec![0.2; 32],
            threshold: DEFAULT_VERIFIER_THRESHOLD,
        };
        assert!(
            !verifier.is_trained(),
            "Mismatched scaler dims should report untrained"
        );

        // Also test partial mismatch: only scaler_std populated.
        let verifier2 = VoiceVerifier {
            trained: true,
            weights: vec![0.5; 96],
            bias: 0.0,
            w1: Vec::new(),
            b1: Vec::new(),
            w2: Vec::new(),
            b2: Vec::new(),
            w3: Vec::new(),
            b3: 0.0,
            scaler_mean: Vec::new(),
            scaler_std: vec![0.2; 32], // non-empty but mismatched
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
        let score = deserialized.predict(&[0.0; EMBEDDING_DIM]);
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
        assert_eq!(verifier.w1.len(), EMBEDDING_DIM * MLP_HIDDEN_1);
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

        // Known 96-dim input.
        let input: Vec<f32> = (0..EMBEDDING_DIM).map(|i| i as f32 * 0.01).collect();
        // Known weights and bias.
        let weights: Vec<f32> = (0..EMBEDDING_DIM)
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

        // ── With scaler ───────────────────────────────────────────────
        let scaler_mean: Vec<f32> = (0..EMBEDDING_DIM).map(|i| (i as f32).sin() * 0.1).collect();
        let scaler_std: Vec<f32> = (0..EMBEDDING_DIM)
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
            verifier_scaled.is_trained(),
            "Legacy linear verifier (with scaler) must report trained",
        );

        // Compute expected with scaling: z = sigmoid(dot(standardize(x), w) + b)
        let scaled: Vec<f32> = input
            .iter()
            .zip(verifier_scaled.scaler_mean.iter())
            .zip(verifier_scaled.scaler_std.iter())
            .map(|((&val, &mean), &std)| if std > 0.0 { (val - mean) / std } else { val })
            .collect();
        let dot_scaled: f32 = scaled.iter().zip(weights.iter()).map(|(x, w)| x * w).sum();
        let expected_scaled = sigmoid(dot_scaled + bias);
        let actual_scaled = verifier_scaled.predict(&input);
        assert!(
            (actual_scaled - expected_scaled).abs() < 1e-5,
            "Legacy linear (with scaler): expected {expected_scaled:.6}, got {actual_scaled:.6}",
        );
    }

    #[test]
    fn test_backward_compat_linear_wrong_weights_dim_detected() {
        // A legacy linear verifier with wrong-length weights must be rejected
        // by is_trained() — wrong dimensions would produce truncated dot
        // products via zip() in predict().
        let verifier = VoiceVerifier {
            trained: true,
            weights: vec![0.5; 32], // wrong: 32 ≠ 96
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
            "Legacy linear verifier with wrong weights dim (32) must report untrained",
        );

        // Also verify that correct-length weights still passes.
        let verifier_ok = VoiceVerifier {
            weights: vec![0.5; 96],
            ..verifier
        };
        assert!(
            verifier_ok.is_trained(),
            "Legacy linear verifier with correct weights dim (96) must report trained",
        );
    }

    #[test]
    fn test_backward_compat_corrupted_mlp_falls_back_to_linear() {
        // When MLP parameters exist but have wrong dimensions (corrupted
        // serialization), is_trained() should accept the linear fallback
        // and predict() should use weights/bias instead of mlp_forward().

        let input: Vec<f32> = (0..EMBEDDING_DIM).map(|i| i as f32 * 0.01).collect();

        // Corrupted MLP: w1 has wrong length (half the correct size).
        let verifier = VoiceVerifier {
            trained: true,
            weights: vec![0.1; EMBEDDING_DIM], // valid linear weights
            bias: 0.5,
            w1: vec![0.2; EMBEDDING_DIM * MLP_HIDDEN_1 / 2], // wrong length
            b1: vec![0.3; MLP_HIDDEN_1],
            w2: vec![0.4; MLP_HIDDEN_1 * MLP_HIDDEN_2],
            b2: vec![0.5; MLP_HIDDEN_2],
            w3: vec![0.6; MLP_HIDDEN_2],
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
            w1: vec![0.2; EMBEDDING_DIM * MLP_HIDDEN_1], // now correct length
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

            // Apply scaler once (same logic as predict()).
            let x: Vec<f32> = if !verifier.scaler_mean.is_empty() && !verifier.scaler_std.is_empty()
            {
                emb.iter()
                    .zip(verifier.scaler_mean.iter())
                    .zip(verifier.scaler_std.iter())
                    .map(
                        |((&val, &mean), &std)| {
                            if std > 0.0 { (val - mean) / std } else { val }
                        },
                    )
                    .collect()
            } else {
                emb.clone()
            };

            // predict() applies scaler internally + routes to MLP.
            // predict_scaled() skips scaler + routes to MLP directly.
            // Both must produce the same output (scaler consistency).
            let from_predict = verifier.predict(&emb);
            let from_scaled = predict_scaled(&verifier, &x);
            assert!(
                (from_predict - from_scaled).abs() < 1e-5,
                "predict()={from_predict:.6} ≠ predict_scaled()={from_scaled:.6} \
                 — scaler routing mismatch",
            );

            // predict_scaled() must match direct mlp_forward() call
            // (routing correctness, not a tautology since predict_scaled
            //  uses the same has_valid_mlp_params() logic as predict()).
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
            &[vec![0.0; 96]],
            None,
            DEFAULT_VERIFIER_THRESHOLD,
            0.001,
            0.1,
            100,
        );
        assert!(!verifier.is_trained());
    }
}
