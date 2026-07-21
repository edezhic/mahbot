//! Logistic regression verifier for wake word false-trigger suppression.
//!
//! Implements a lightweight second-stage classifier that runs AFTER DTW
//! matching fires, as an additional AND gate. The verifier uses L2-regularized
//! logistic regression with optional StandardScaler normalization.
//!
//! When not trained, the verifier acts as a no-op (all frames pass).
//!
//! # Architecture
//!
//! Training uses batch gradient descent on positive (enrollment) and negative
//! (synthetic or real) embedding examples. Inference is a single dot product
//! plus sigmoid — ~1μs per frame.
//!
//! ## Training data
//!
//! - **Positive examples**: Mean-pooled 96-dim embeddings from each enrollment
//!   utterance (10 per enrollment).
//! - **Negative examples**: Synthetic Gaussian noise (bootstrapping) or
//!   hard-negative embeddings collected from near-miss frames during detection.

use serde::{Deserialize, Serialize};
use tracing::warn;

/// Default decision threshold for the verifier.
///
/// Lowered from 0.5 to 0.3 (mahbot-788) to reduce false negatives. The DTW
/// matching already provides the primary false-trigger protection, so the
/// verifier threshold can be more permissive.
const DEFAULT_VERIFIER_THRESHOLD: f32 = 0.3;

/// L2 regularization strength (lambda).
///
/// Chosen for stability with the default learning rate (0.01):
/// `lr * λ = 0.01`, giving a weight decay factor of 0.99 per gradient step.
/// The steady-state weights satisfy `w_j = -g_j / λ` where `g_j` is the data
/// gradient, so λ = 1.0 provides moderate regularization that prevents
/// overfitting on the small (~10) enrollment samples without dominating the
/// data signal.
const L2_LAMBDA: f32 = 1.0;

/// Learning rate for gradient descent.
const LEARNING_RATE: f32 = 0.01;

/// Maximum iterations for gradient descent.
const MAX_ITER: usize = 2000;

/// Embedding dimensionality (used by both verifier and voice pipeline).
pub(crate) const EMBEDDING_DIM: usize = 96;

/// Number of synthetic negative examples to generate for bootstrapping
/// when no real calibration data is available.
const SYNTHETIC_NEGATIVES_COUNT: usize = 30;

// ═══════════════════════════════════════════════════════════════════════════
// VoiceVerifier
// ═══════════════════════════════════════════════════════════════════════════

/// A lightweight logistic regression verifier for wake word false-trigger
/// suppression (mahbot-777).
///
/// Computes `sigmoid(w·x + b)` for a given 96-dim embedding, optionally
/// normalized with a StandardScaler. If the score is below `threshold`, the
/// wake word detection is suppressed.
///
/// When `trained` is `false`, the verifier is a no-op (all frames pass with
/// score 1.0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceVerifier {
    /// 96-dim weight vector for the logistic regression.
    #[serde(default)]
    pub weights: Vec<f32>,
    /// Bias term.
    #[serde(default)]
    pub bias: f32,
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
            scaler_mean: Vec::new(),
            scaler_std: Vec::new(),
            threshold: DEFAULT_VERIFIER_THRESHOLD,
            trained: false,
        }
    }

    /// Returns `true` if this verifier has been trained and is ready for
    /// inference.
    ///
    /// Validates that scaler dimensions match weights when scalers are
    /// populated, preventing silently wrong predictions from corrupted
    /// deserialization.
    #[must_use]
    pub fn is_trained(&self) -> bool {
        if !self.trained || self.weights.is_empty() {
            return false;
        }
        // If either scaler is non-empty, both must be present and match the
        // weight dimensionality.
        if (!self.scaler_mean.is_empty() || !self.scaler_std.is_empty())
            && (self.scaler_mean.len() != self.weights.len()
                || self.scaler_std.len() != self.weights.len())
        {
            return false;
        }
        true
    }

    /// Predict the probability that the given 96-dim embedding is a genuine
    /// wake word.
    ///
    /// Returns a score in `[0.0, 1.0]`. When untrained, always returns `1.0`
    /// (no-op — all frames pass).
    #[must_use]
    pub fn predict(&self, embedding: &[f32]) -> f32 {
        if !self.is_trained() {
            return 1.0;
        }

        // Validate embedding dimension matches weights. A mismatch would
        // silently truncate via zip, producing wrong results.
        if embedding.len() != self.weights.len() {
            warn!(
                "Verifier embedding dimension mismatch: got {}, expected {}; falling back to no-op",
                embedding.len(),
                self.weights.len(),
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

        // Linear combination: z = w·x + b
        let z: f32 = x
            .iter()
            .zip(self.weights.iter())
            .map(|(x, w)| x * w)
            .sum::<f32>()
            + self.bias;

        // Sigmoid activation
        sigmoid(z)
    }

    /// Train a new verifier from positive and negative 96-dim embedding
    /// examples using L2-regularized logistic regression.
    ///
    /// # Arguments
    ///
    /// * `positive_embeddings` — Mean-pooled embeddings from enrollment
    ///   utterances (label = 1). Each element is a single 96-dim vector.
    /// * `negative_embeddings` — Embeddings from non-wake-word audio
    ///   (label = 0). Each element is a single 96-dim vector.
    /// * `threshold` — Decision threshold (typically 0.5).
    /// * `l2_lambda` — L2 regularisation strength.
    /// * `learning_rate` — Gradient descent learning rate.
    /// * `max_iter` — Maximum gradient descent iterations.
    ///
    /// Returns a trained `VoiceVerifier`, or an untrained verifier if either
    /// input list is empty.
    #[must_use]
    pub fn train(
        positive_embeddings: &[Vec<f32>],
        negative_embeddings: &[Vec<f32>],
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

        // 3. Train logistic regression on scaled features
        let (weights, bias) = train_logistic_regression(
            &scaled_features,
            &labels,
            dim,
            l2_lambda,
            learning_rate,
            max_iter,
        );

        Self {
            weights,
            bias,
            scaler_mean,
            scaler_std,
            threshold,
            trained: true,
        }
    }

    /// Convenience: train a verifier using the given positive embeddings and
    /// automatically generated synthetic negative examples (Gaussian noise).
    #[must_use]
    pub fn train_with_synthetic_negatives(
        positive_embeddings: &[Vec<f32>],
        threshold: f32,
    ) -> Self {
        let dim = positive_embeddings
            .first()
            .map_or(EMBEDDING_DIM, std::vec::Vec::len);
        let negatives = generate_synthetic_negatives(SYNTHETIC_NEGATIVES_COUNT, dim);
        Self::train(
            positive_embeddings,
            &negatives,
            threshold,
            L2_LAMBDA,
            LEARNING_RATE,
            MAX_ITER,
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
// Logistic regression training
// ═══════════════════════════════════════════════════════════════════════════

/// Train a logistic regression model using batch gradient descent with L2
/// regularisation.
///
/// The cross-entropy loss with L2 penalty is:
/// ```text
/// J(w) = -(1/N) Σ [y·log(σ) + (1-y)·log(1-σ)] + (λ/2)·||w||²
/// ```
///
/// Gradient (averaged over the batch):
/// ```text
/// ∂J/∂wⱼ = (1/N) Σ (σ_i - y_i)·x_ij + λ·wⱼ
/// ∂J/∂b  = (1/N) Σ (σ_i - y_i)
/// ```
///
/// Returns `(weights, bias)` where `weights` has length `dim`.
fn train_logistic_regression(
    features: &[Vec<f32>],
    labels: &[f32],
    dim: usize,
    l2_lambda: f32,
    learning_rate: f32,
    max_iter: usize,
) -> (Vec<f32>, f32) {
    let n = features.len();
    if n == 0 || dim == 0 {
        return (Vec::new(), 0.0);
    }

    let mut weights = vec![0.0; dim];
    let mut bias = 0.0;
    #[allow(clippy::cast_precision_loss)]
    let n_f32 = n as f32;

    for _iteration in 0..max_iter {
        // ── Compute batch gradients ──
        let mut dw = vec![0.0; dim];
        let mut db = 0.0;

        for i in 0..n {
            // Linear combination
            let z: f32 = features[i]
                .iter()
                .zip(weights.iter())
                .map(|(x, w)| x * w)
                .sum::<f32>()
                + bias;
            let pred = sigmoid(z);
            let error = pred - labels[i]; // (σ - y) for cross-entropy gradient

            for j in 0..dim {
                dw[j] += error * features[i][j];
            }
            db += error;
        }

        // Average over the batch and add L2 regularisation (only for weights,
        // not the bias term — matching sklearn convention).
        for j in 0..dim {
            dw[j] = dw[j] / n_f32 + l2_lambda * weights[j];
        }
        db /= n_f32;

        // ── Gradient descent update ──
        for j in 0..dim {
            weights[j] -= learning_rate * dw[j];
        }
        bias -= learning_rate * db;
    }

    (weights, bias)
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
/// preserved as a utility for potential future use.
#[must_use]
#[allow(dead_code)]
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

    #[test]
    fn test_verifier_accepts_positive() {
        // Train on known positive and negative synthetic embeddings.
        let mut rng = StdRng::seed_from_u64(42);
        let positives: Vec<Vec<f32>> = (0..20).map(|_| make_positive_embedding(&mut rng)).collect();
        let negatives: Vec<Vec<f32>> = (0..30).map(|_| make_negative_embedding(&mut rng)).collect();

        let verifier = VoiceVerifier::train(
            &positives, &negatives, 0.5,   // threshold
            0.001, // weak L2 (clean synthetic data)
            0.1,   // learning rate
            500,   // max iter
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

        let verifier = VoiceVerifier::train(&positives, &negatives, 0.5, 0.001, 0.1, 500);

        assert!(verifier.is_trained());

        // Verify a held-out negative is rejected.
        let held_out = make_negative_embedding(&mut rng);
        let score = verifier.predict(&held_out);
        assert!(
            score < 0.5,
            "Verifier should reject negative embedding (score < 0.5), got score={score:.4}",
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

        let verifier = VoiceVerifier::train(&positives, &negatives, 0.5, 0.001, 0.1, 500);

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
            scaler_mean: vec![0.1; 32], // wrong dimension (32 ≠ 96)
            scaler_std: vec![0.2; 32],
            threshold: 0.5,
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
            scaler_mean: Vec::new(),
            scaler_std: vec![0.2; 32], // non-empty but mismatched
            threshold: 0.5,
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
        let positives: Vec<Vec<f32>> = (0..10).map(|_| make_positive_embedding(&mut rng)).collect();
        let verifier = VoiceVerifier::train_with_synthetic_negatives(&positives, 0.5);
        assert!(verifier.is_trained());
        assert_eq!(verifier.weights.len(), EMBEDDING_DIM);
        assert!(!verifier.scaler_mean.is_empty());
        assert!(!verifier.scaler_std.is_empty());

        // All weights must be finite — NaN/inf indicates gradient divergence
        // from unstable hyperparameters.
        for (j, &w) in verifier.weights.iter().enumerate() {
            assert!(
                w.is_finite(),
                "Weight[{j}] is not finite: {w}; gradient descent diverged",
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
    fn test_verifier_empty_training_returns_untrained() {
        // No positive examples → should return untrained.
        let verifier = VoiceVerifier::train(&[], &[vec![0.0; 96]], 0.5, 0.001, 0.1, 100);
        assert!(!verifier.is_trained());
    }
}
