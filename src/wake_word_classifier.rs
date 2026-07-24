//! Conv1D wake word classifier — replaces DTW template matching.
//!
//! Architecture: Conv1D(96→64, k=3) + BN + ReLU → Conv1D(64→64, k=3) + BN + ReLU
//! → AdaptiveAvgPool1d → Linear(64→1) + Sigmoid.
//!
//! Inference uses pure Rust. Training uses manual backprop + Adam.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use anyhow::Result;
use rand::RngExt;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::voice_verifier::EMBEDDING_DIM;

// ── Constants ────────────────────────────────────────────────────────────

pub const WINDOW_SIZE: usize = 3;

/// Number of ensemble members to train for wake word detection (mahbot-839).
/// Five independent models with different seeds are trained during enrollment
/// and their post-sigmoid scores are averaged at inference time.
pub const NUM_ENSEMBLE_MEMBERS: usize = 5;
pub const INPUT_DIM: usize = WINDOW_SIZE * EMBEDDING_DIM; // 288
const CONV1_OUT: usize = 64;
const CONV2_OUT: usize = 64;
const KERNEL_SIZE: usize = 3;
const PADDING: usize = 1;
const FC_OUT: usize = 1;
/// L2 regularization strength (lambda).
///
/// Set to 0.0001 (mahbot-835).  The 829 baseline of 0.001 caused the
/// Conv1D classifier to underfit with the expanded negative dataset
/// (2093 embeddings from 20 confusable + 20 unrelated × 3 seeds).
/// Even at 0.0005 the classifier still produced sub-0.80 per-frame
/// scores, failing the rolling window gate.  At 0.0001 the model can
/// separate wake word frames from the 10× larger negative set while
/// MATCH_THRESHOLD_FACTOR prevents false accepts.
const L2_LAMBDA: f32 = 0.0001;
const LEARNING_RATE: f32 = 0.001;
const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;
const BATCH_SZ: usize = 32;
const MAX_EPOCHS: usize = 100;
/// Early stop patience increased from 5 to 15 (mahbot-846) — with small
/// enrollment datasets (~99 positive windows) and a noisy validation signal,
/// models on viable trajectories were getting killed before converging.
/// Many seeds terminated at epoch 10-12 before the optimizer could escape
/// initialization.
const EARLY_STOP_PATIENCE: usize = 15;
const VALIDATION_SPLIT: f32 = 0.2;

/// Dropout probability applied after each ReLU during training (mahbot-846).
/// The model has ~31K trainable parameters trained on ~99 positive examples
/// with only weak L2 weight decay (λ=0.0001).  Dropout provides stronger
/// regularization to prevent overfitting on small enrollment data.
const DROPOUT_RATE: f32 = 0.3;

/// Maximum gradient norm for clipping (mahbot-846).  With small datasets,
/// a single unlucky mini-batch can produce large gradients that derail
/// the optimization trajectory.  Gradient clipping caps the L2 norm of
/// the flattened gradient vector to this value before the Adam update.
const GRADIENT_CLIP_NORM: f32 = 1.0;

/// Momentum for updating batch-norm running statistics (mahbot-846).
/// During training, each forward pass computes batch mean/variance over
/// the spatial dimension and updates running stats via exponential moving
/// average: running = momentum * running + (1 - momentum) * batch.
const BN_MOMENTUM: f32 = 0.9;

/// Standard deviation of Gaussian noise applied to embedding windows during
/// training for data augmentation (mahbot-847).  Noise is added to the
/// L2-normalized 288-dim window, then the result is re-normalized.
/// Calibrated to produce meaningful variation (cosine similarity ~0.76
/// after augmentation) without destroying the wake word signal.
pub const DATA_AUGMENTATION_STD: f32 = 0.05;

// ── Weights ─────────────────────────────────────────────────────────────

/// Default running mean for BN1 (used when deserializing legacy enrollment
/// data that predates the BN training stats, preserving backward compatibility).
fn default_bn1_running_mean() -> Vec<f32> {
    vec![0.0; CONV1_OUT]
}

/// Default running variance for BN1 (see [`default_bn1_running_mean`]).
fn default_bn1_running_var() -> Vec<f32> {
    vec![1.0; CONV1_OUT]
}

/// Default running mean for BN2 (see [`default_bn1_running_mean`]).
fn default_bn2_running_mean() -> Vec<f32> {
    vec![0.0; CONV2_OUT]
}

/// Default running variance for BN2 (see [`default_bn1_running_mean`]).
fn default_bn2_running_var() -> Vec<f32> {
    vec![1.0; CONV2_OUT]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierWeights {
    pub conv1_weight: Vec<f32>, // [64, 96, 3]
    pub conv1_bias: Vec<f32>,   // [64]
    pub bn1_gamma: Vec<f32>,
    pub bn1_beta: Vec<f32>,
    #[serde(default = "default_bn1_running_mean")]
    pub bn1_running_mean: Vec<f32>,
    #[serde(default = "default_bn1_running_var")]
    pub bn1_running_var: Vec<f32>,
    pub conv2_weight: Vec<f32>, // [64, 64, 3]
    pub conv2_bias: Vec<f32>,   // [64]
    pub bn2_gamma: Vec<f32>,
    pub bn2_beta: Vec<f32>,
    #[serde(default = "default_bn2_running_mean")]
    pub bn2_running_mean: Vec<f32>,
    #[serde(default = "default_bn2_running_var")]
    pub bn2_running_var: Vec<f32>,
    pub fc_weight: Vec<f32>, // [1, 64]
    pub fc_bias: Vec<f32>,   // [1]
    pub bn_eps: f32,
}

impl Default for ClassifierWeights {
    fn default() -> Self {
        Self::from_rng(&mut rand::rng())
    }
}

impl ClassifierWeights {
    /// Initialize classifier weights using a seeded RNG for deterministic training.
    /// Used by [`train_classifier`] when a seed is configured, replacing the
    /// non-deterministic `Default` path.
    pub fn from_rng(rng: &mut (impl rand::Rng + ?Sized)) -> Self {
        // Xavier/Glorot uniform initialization corrected for Conv1D
        // (mahbot-846): fan_in and fan_out must include kernel_size.
        // Formula: scale = sqrt(6 / (fan_in + fan_out))
        //   fan_in = in_channels * kernel_size
        //   fan_out = out_channels * kernel_size
        let scale_c1 = (6.0 / ((EMBEDDING_DIM + CONV1_OUT) * KERNEL_SIZE) as f32).sqrt();
        let scale_c2 = (6.0 / ((CONV1_OUT + CONV2_OUT) * KERNEL_SIZE) as f32).sqrt();
        let scale_fc = (6.0 / (CONV2_OUT + FC_OUT) as f32).sqrt();
        let mut uniform =
            |s: f32, n: usize| -> Vec<f32> { (0..n).map(|_| rng.random_range(-s..s)).collect() };
        Self {
            conv1_weight: uniform(scale_c1, CONV1_OUT * EMBEDDING_DIM * KERNEL_SIZE),
            conv1_bias: vec![0.0; CONV1_OUT],
            bn1_gamma: vec![1.0; CONV1_OUT],
            bn1_beta: vec![0.0; CONV1_OUT],
            bn1_running_mean: vec![0.0; CONV1_OUT],
            bn1_running_var: vec![1.0; CONV1_OUT],
            conv2_weight: uniform(scale_c2, CONV2_OUT * CONV1_OUT * KERNEL_SIZE),
            conv2_bias: vec![0.0; CONV2_OUT],
            bn2_gamma: vec![1.0; CONV2_OUT],
            bn2_beta: vec![0.0; CONV2_OUT],
            bn2_running_mean: vec![0.0; CONV2_OUT],
            bn2_running_var: vec![1.0; CONV2_OUT],
            fc_weight: uniform(scale_fc, CONV2_OUT * FC_OUT),
            fc_bias: vec![0.0; FC_OUT],
            bn_eps: 1e-5,
        }
    }

    /// Return references to all weight Vec fields for unified validation/counting.
    /// Adding a new Vec field to `ClassifierWeights` requires adding it here
    /// — a compile error if omitted from the array literal (Rust checks array
    /// element count at compile time for fixed-size arrays).
    pub(crate) fn all_weight_slices(&self) -> [&[f32]; 14] {
        [
            &self.conv1_weight,
            &self.conv1_bias,
            &self.bn1_gamma,
            &self.bn1_beta,
            &self.bn1_running_mean,
            &self.bn1_running_var,
            &self.conv2_weight,
            &self.conv2_bias,
            &self.bn2_gamma,
            &self.bn2_beta,
            &self.bn2_running_mean,
            &self.bn2_running_var,
            &self.fc_weight,
            &self.fc_bias,
        ]
    }

    /// Return references to all trainable (optimizable) `Vec` fields, excluding
    /// non-trainable batch-norm running statistics.  Used for `param_count()` and
    /// degenerate-solution checks where including non-trainable stats (which have
    /// different learned-vs-initialization dynamics) would misrepresent the actual
    /// parameter count or trigger false positives.
    pub(crate) fn all_trainable_slices(&self) -> [&[f32]; 10] {
        [
            &self.conv1_weight,
            &self.conv1_bias,
            &self.bn1_gamma,
            &self.bn1_beta,
            &self.conv2_weight,
            &self.conv2_bias,
            &self.bn2_gamma,
            &self.bn2_beta,
            &self.fc_weight,
            &self.fc_bias,
        ]
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.conv1_weight.len() == CONV1_OUT * EMBEDDING_DIM * KERNEL_SIZE);
        anyhow::ensure!(self.conv1_bias.len() == CONV1_OUT);
        anyhow::ensure!(self.bn1_gamma.len() == CONV1_OUT);
        anyhow::ensure!(self.bn1_beta.len() == CONV1_OUT);
        anyhow::ensure!(self.bn1_running_mean.len() == CONV1_OUT);
        anyhow::ensure!(self.bn1_running_var.len() == CONV1_OUT);
        anyhow::ensure!(self.conv2_weight.len() == CONV2_OUT * CONV1_OUT * KERNEL_SIZE);
        anyhow::ensure!(self.conv2_bias.len() == CONV2_OUT);
        anyhow::ensure!(self.bn2_gamma.len() == CONV2_OUT);
        anyhow::ensure!(self.bn2_beta.len() == CONV2_OUT);
        anyhow::ensure!(self.bn2_running_mean.len() == CONV2_OUT);
        anyhow::ensure!(self.bn2_running_var.len() == CONV2_OUT);
        anyhow::ensure!(self.fc_weight.len() == CONV2_OUT * FC_OUT);
        anyhow::ensure!(self.fc_bias.len() == FC_OUT);
        // Check for NaN/Infinity — guards against silent training failures
        // (NaN gradients, degenerate input normalization) that shape checks
        // alone don't catch.
        anyhow::ensure!(
            self.all_weight_slices()
                .iter()
                .flat_map(|s| s.iter())
                .all(|v| v.is_finite()),
            "Classifier weights contain NaN or Infinity"
        );
        Ok(())
    }
    pub fn param_count(&self) -> usize {
        self.all_trainable_slices().iter().map(|s| s.len()).sum()
    }
}

// ── Classifier ──────────────────────────────────────────────────────────

pub struct WakeWordClassifier {
    /// Ensemble member weight sets.  In single-model mode this contains one
    /// entry; in ensemble mode (mahbot-839) it contains `NUM_ENSEMBLE_MEMBERS`
    /// entries and `forward()` averages their post-sigmoid scores.
    members: Vec<ClassifierWeights>,
    /// Per-member validation losses for softmax-weighted averaging (mahbot-847).
    /// When empty (backward compat / legacy single-model path), falls back
    /// to uniform averaging.
    member_val_losses: Vec<f32>,
    /// Pre-computed softmax weights cached at construction time (mahbot-847).
    /// Avoids recomputing on every inference frame.
    cached_weights: Vec<f32>,
}

// ── Training context ────────────────────────────────────────────────────

/// Per-sample training context passed between forward and backward passes
/// during classifier training (mahbot-846).
///
/// Stores batch-normalization statistics (computed from the spatial
/// dimension in the forward pass and consumed by the backward pass) and
/// dropout masks.  A new context is created for each training sample.
struct TrainingCtx {
    bn1_mean: Vec<f32>,
    bn1_var: Vec<f32>,
    bn2_mean: Vec<f32>,
    bn2_var: Vec<f32>,
    dropout_mask1: Vec<f32>,
    dropout_mask2: Vec<f32>,
    rng: rand::rngs::StdRng,
}

impl TrainingCtx {
    fn new(seed: u64) -> Self {
        Self {
            bn1_mean: vec![0.0; CONV1_OUT],
            bn1_var: vec![0.0; CONV1_OUT],
            bn2_mean: vec![0.0; CONV2_OUT],
            bn2_var: vec![0.0; CONV2_OUT],
            dropout_mask1: vec![0.0; CONV1_OUT * WINDOW_SIZE],
            dropout_mask2: vec![0.0; CONV2_OUT * WINDOW_SIZE],
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }
}

/// Compute per-channel mean and variance over the spatial dimension for a
/// Conv1D output of shape `[c, l]` in channels-first layout.
fn compute_batch_stats(x: &[f32], c: usize, l: usize, mean: &mut [f32], var: &mut [f32]) {
    for ci in 0..c {
        let mut m = 0.0;
        for li in 0..l {
            m += x[ci * l + li];
        }
        m /= l as f32;
        mean[ci] = m;
        let mut v = 0.0;
        for li in 0..l {
            let d = x[ci * l + li] - m;
            v += d * d;
        }
        v /= l as f32;
        var[ci] = v;
    }
}

/// Apply dropout with inverted scaling (mahbot-846).  Sets dropped elements
/// to zero and scales kept elements by 1/(1-rate).  The mask stores the
/// scale (either 0.0 or 1/(1-rate)) for use in the backward pass.
fn apply_dropout(x: &mut [f32], rate: f32, mask: &mut [f32], rng: &mut impl rand::Rng) {
    let scale = 1.0 / (1.0 - rate);
    for (v, m) in x.iter_mut().zip(mask.iter_mut()) {
        if rng.random::<f32>() < rate {
            *v = 0.0;
            *m = 0.0;
        } else {
            *v *= scale;
            *m = scale;
        }
    }
}

/// Apply data augmentation to an L2-normalized embedding window (mahbot-847).
/// Adds Gaussian noise with the given standard deviation and re-normalizes
/// the perturbed vector back to unit length.  Uses the provided RNG for
/// deterministic noise generation.
fn apply_augmentation(x: &[f32], std: f32, rng: &mut impl rand::Rng) -> Vec<f32> {
    let mut out = x.to_vec();
    // Box-Muller transform for Gaussian noise
    for v in &mut out {
        let u1 = rng.random::<f32>();
        let u2 = rng.random::<f32>();
        // Guard against log(0) — clamp u1 away from 0.
        let u1_safe = (1.0 - u1).max(f32::EPSILON);
        let z = (-2.0 * u1_safe.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        *v += std * z;
    }
    // Re-normalize to unit length
    let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
    for v in &mut out {
        *v /= norm;
    }
    out
}

// ── Training forward pass ─────────────────────────────────────────────

/// Inference-only forward pass (no mutation of weights).  Uses running
/// batch-norm statistics and no dropout — identical to the original
/// `forward_pass` behaviour.
fn forward_pass_infer(x: &[f32], w: &ClassifierWeights) -> f32 {
    let mut h = conv1d(
        x,
        EMBEDDING_DIM,
        WINDOW_SIZE,
        CONV1_OUT,
        &w.conv1_weight,
        &w.conv1_bias,
    );
    batch_norm(
        &mut h,
        CONV1_OUT,
        WINDOW_SIZE,
        &w.bn1_gamma,
        &w.bn1_beta,
        &w.bn1_running_mean,
        &w.bn1_running_var,
        w.bn_eps,
    );
    relu(&mut h);
    let mut h = conv1d(
        &h,
        CONV1_OUT,
        WINDOW_SIZE,
        CONV2_OUT,
        &w.conv2_weight,
        &w.conv2_bias,
    );
    batch_norm(
        &mut h,
        CONV2_OUT,
        WINDOW_SIZE,
        &w.bn2_gamma,
        &w.bn2_beta,
        &w.bn2_running_mean,
        &w.bn2_running_var,
        w.bn_eps,
    );
    relu(&mut h);
    let pooled = adaptive_avg_pool(&h, CONV2_OUT, WINDOW_SIZE);
    sigmoid(dot(&pooled, &w.fc_weight) + w.fc_bias[0])
}

/// Training forward pass with batch normalisation, running stats
/// collection, and dropout (mahbot-846).
///
/// - Batch norm uses per-sample batch statistics (mean/variance over the
///   spatial dimension) instead of fixed running statistics.
/// - Per-sample mean/variance are recorded in `ctx` and accumulated by the
///   caller for a single batch-level running stats update (avoids compounding
///   the momentum decay across samples within a batch).
/// - Dropout (rate [`DROPOUT_RATE`]) is applied after each ReLU.
/// - The context records batch stats and dropout masks for the backward pass.
///
/// # Backward note
/// The backward pass uses a simplified gradient that treats the BN statistics
/// as constants (no backprop through the mean/variance computation).  This
/// approximation is common in resource-constrained settings and converges
/// reliably in practice, but differs from the full BN backward that includes
/// the mean/variance correction terms.
fn forward_pass_train(x: &[f32], w: &ClassifierWeights, ctx: &mut TrainingCtx) -> f32 {
    let mut h = conv1d(
        x,
        EMBEDDING_DIM,
        WINDOW_SIZE,
        CONV1_OUT,
        &w.conv1_weight,
        &w.conv1_bias,
    );

    // ── BN1 with batch stats ──
    compute_batch_stats(
        &h,
        CONV1_OUT,
        WINDOW_SIZE,
        &mut ctx.bn1_mean,
        &mut ctx.bn1_var,
    );
    batch_norm(
        &mut h,
        CONV1_OUT,
        WINDOW_SIZE,
        &w.bn1_gamma,
        &w.bn1_beta,
        &ctx.bn1_mean,
        &ctx.bn1_var,
        w.bn_eps,
    );
    relu(&mut h);
    apply_dropout(&mut h, DROPOUT_RATE, &mut ctx.dropout_mask1, &mut ctx.rng);

    // ── Conv2 ──
    let mut h = conv1d(
        &h,
        CONV1_OUT,
        WINDOW_SIZE,
        CONV2_OUT,
        &w.conv2_weight,
        &w.conv2_bias,
    );

    // ── BN2 with batch stats ──
    compute_batch_stats(
        &h,
        CONV2_OUT,
        WINDOW_SIZE,
        &mut ctx.bn2_mean,
        &mut ctx.bn2_var,
    );
    batch_norm(
        &mut h,
        CONV2_OUT,
        WINDOW_SIZE,
        &w.bn2_gamma,
        &w.bn2_beta,
        &ctx.bn2_mean,
        &ctx.bn2_var,
        w.bn_eps,
    );
    relu(&mut h);
    apply_dropout(&mut h, DROPOUT_RATE, &mut ctx.dropout_mask2, &mut ctx.rng);

    let pooled = adaptive_avg_pool(&h, CONV2_OUT, WINDOW_SIZE);
    sigmoid(dot(&pooled, &w.fc_weight) + w.fc_bias[0])
}

impl WakeWordClassifier {
    /// Create a single-member classifier (legacy / backward compat path).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new(weights: ClassifierWeights) -> Self {
        Self {
            members: vec![weights],
            member_val_losses: vec![],
            cached_weights: vec![1.0],
        }
    }

    /// Return a reference to the ensemble member weights.
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub fn weights_ref(&self) -> &[ClassifierWeights] {
        &self.members
    }

    /// Return a reference to the per-member validation losses (empty if
    /// unavailable, e.g. backward compat / uniform-averaging mode).
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub fn val_losses_ref(&self) -> &[f32] {
        &self.member_val_losses
    }

    /// Create a multi-member ensemble classifier.
    ///
    /// # Panics
    /// Panics if `members` is empty — an ensemble must have at least one model.
    pub fn new_ensemble(members: Vec<ClassifierWeights>) -> Self {
        assert!(
            !members.is_empty(),
            "Ensemble must have at least one member"
        );
        let n = members.len();
        Self {
            members,
            member_val_losses: vec![],
            cached_weights: vec![1.0 / n as f32; n],
        }
    }

    /// Create a multi-member ensemble classifier with per-member validation
    /// losses for softmax-weighted averaging (mahbot-847).
    ///
    /// When `val_losses` has the wrong length or is empty, falls back to
    /// uniform averaging (backward compatibility).
    ///
    /// # Panics
    /// Panics if `members` is empty — an ensemble must have at least one model.
    pub fn new_ensemble_weighted(members: Vec<ClassifierWeights>, val_losses: Vec<f32>) -> Self {
        assert!(
            !members.is_empty(),
            "Ensemble must have at least one member"
        );
        let n = members.len();
        let (member_val_losses, cached_weights) = if val_losses.len() == n {
            let weights = Self::compute_softmax_weights(&val_losses, n);
            (val_losses, weights)
        } else {
            (vec![], vec![1.0 / n as f32; n])
        };
        Self {
            members,
            member_val_losses,
            cached_weights,
        }
    }

    /// Compute softmax weights from member validation losses.
    /// Uses `exp(-(loss - min_loss))` for numerical stability, ensuring the
    /// best model (lowest loss) always gets the highest weight.
    /// Falls back to uniform weights when validation losses are unavailable.
    fn compute_softmax_weights(val_losses: &[f32], n: usize) -> Vec<f32> {
        // Caller (new_ensemble_weighted) ensures len matches and n > 0.
        debug_assert!(val_losses.len() == n && n > 0);
        // Guard against NaN in validation losses — if any entry is NaN the
        // softmax computation produces all-NaN weights, corrupting inference.
        if val_losses.iter().any(|v| v.is_nan()) {
            return vec![1.0 / n as f32; n];
        }
        let min_loss = val_losses.iter().copied().fold(f32::INFINITY, f32::min);
        let mut exps = Vec::with_capacity(n);
        let mut sum = 0.0;
        for &loss in val_losses {
            let e = (-(loss - min_loss)).exp();
            exps.push(e);
            sum += e;
        }
        // sum is always > 0 for finite inputs, but guard defensively.
        debug_assert!(sum > 0.0, "Sum of exponentials must be positive");
        exps.iter().map(|e| e / sum).collect()
    }

    /// Run the forward pass through all ensemble members and return the
    /// softmax-weighted average post-sigmoid score (mahbot-847).
    ///
    /// For a single-model classifier this is equivalent to the original
    /// single-member forward pass.
    /// For an unweighted ensemble (no val_losses available), falls back to
    /// uniform averaging.
    pub fn forward(&self, embeddings: &[Vec<f32>]) -> f32 {
        debug_assert_eq!(embeddings.len(), WINDOW_SIZE);
        // Flatten 3 embeddings into a 288-dim window, then L2-normalize
        // as a single vector — matching the training data pipeline
        // (train_classifier normalizes each 288-dim training window to
        // unit length).
        let mut x = vec![0.0; EMBEDDING_DIM * WINDOW_SIZE];
        for (t, emb) in embeddings.iter().enumerate() {
            for (c, &v) in emb.iter().enumerate() {
                x[t * EMBEDDING_DIM + c] = v;
            }
        }
        let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
        for v in &mut x {
            *v /= norm;
        }
        // Convert from samples-first to channels-first layout for Conv1D.
        let cf = to_channels_first(&x, EMBEDDING_DIM, WINDOW_SIZE);

        // Softmax-weighted average of post-sigmoid scores across all
        // ensemble members (mahbot-847).  Models with lower validation
        // loss contribute more to the final detection score.
        // Weights are pre-computed at construction time to avoid recomputation
        // on every inference frame.
        debug_assert_eq!(self.cached_weights.len(), self.members.len());
        let mut total = 0.0;
        for (i, w) in self.members.iter().enumerate() {
            total += self.cached_weights[i] * forward_pass_infer(&cf, w);
        }
        total
    }
}

// ── Forward primitives ──────────────────────────────────────────────────

fn conv1d(inp: &[f32], cin: usize, l: usize, cout: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; cout * l];
    for co in 0..cout {
        for li in 0..l {
            let mut s = b[co];
            for ci in 0..cin {
                for k in 0..KERNEL_SIZE {
                    let ii = li as isize + k as isize - PADDING as isize;
                    if ii >= 0 && ii < l as isize {
                        s += inp[ci * l + ii as usize] * w[(co * cin + ci) * KERNEL_SIZE + k];
                    }
                }
            }
            out[co * l + li] = s;
        }
    }
    out
}

fn batch_norm(
    x: &mut [f32],
    c: usize,
    l: usize,
    g: &[f32],
    b: &[f32],
    rm: &[f32],
    rv: &[f32],
    eps: f32,
) {
    for ci in 0..c {
        let std = (rv[ci] + eps).sqrt();
        for li in 0..l {
            let idx = ci * l + li;
            x[idx] = g[ci] * (x[idx] - rm[ci]) / std + b[ci];
        }
    }
}

fn relu(x: &mut [f32]) {
    for v in x {
        *v = v.max(0.0);
    }
}

fn adaptive_avg_pool(x: &[f32], c: usize, l: usize) -> Vec<f32> {
    let mut out = vec![0.0; c];
    for ci in 0..c {
        let mut s = 0.0;
        for li in 0..l {
            s += x[ci * l + li];
        }
        out[ci] = s / l as f32;
    }
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Result of training a wake word classifier, returned by [`train_classifier`].
#[derive(Debug, Clone)]
pub struct ClassifierTrainingResult {
    /// The trained classifier weights.  In ensemble mode (mahbot-839) this
    /// contains all ensemble members' weight sets.
    pub weights: Vec<ClassifierWeights>,
    /// Actual number of epochs trained (may be less than
    /// `TrainingConfig::max_epochs` due to early stopping).
    pub epochs_trained: usize,
    /// Best validation loss achieved during training.
    pub best_val_loss: f32,
    /// Per-member validation losses for softmax-weighted averaging (mahbot-847).
    /// One entry per member in `weights`.  When empty, the caller should fall
    /// back to uniform averaging.
    pub val_losses: Vec<f32>,
    /// Mean positive class score after training.
    #[allow(dead_code)]
    pub pos_scores_mean: f32,
    /// Minimum positive class score after training.
    #[allow(dead_code)]
    pub pos_scores_min: f32,
    /// Maximum positive class score after training.
    #[allow(dead_code)]
    pub pos_scores_max: f32,
    /// Mean negative class score after training.
    #[allow(dead_code)]
    pub neg_scores_mean: f32,
    /// Minimum negative class score after training.
    #[allow(dead_code)]
    pub neg_scores_min: f32,
    /// Maximum negative class score after training.
    #[allow(dead_code)]
    pub neg_scores_max: f32,
}

// ── Training ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub l2_lambda: f32,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub early_stop_patience: usize,
    pub validation_split: f32,
    /// Optional RNG seed for deterministic training.  When `None`, uses
    /// `rand::rng()` (non-deterministic).  When `Some(seed)`, uses
    /// `StdRng::seed_from_u64(seed)` for weight init and data shuffling.
    pub rng_seed: Option<u64>,
    /// Number of folds for stratified k-fold cross-validation (mahbot-847).
    /// When 0 (default), uses the random [`validation_split`] ratio instead.
    /// When > 0, each call with a different [`k_fold_index`] gets a different
    /// fold as validation set while preserving class ratios in each fold.
    pub k_fold_total: usize,
    /// Which fold (0-based) to use as the validation set.
    /// Only meaningful when [`k_fold_total`] > 0.
    pub k_fold_index: usize,
    /// Standard deviation of Gaussian noise for on-the-fly data augmentation
    /// applied to L2-normalized training windows (mahbot-847).
    /// When 0.0 (default), no augmentation is applied.
    /// Typical values: 0.01–0.05.
    pub data_augmentation_std: f32,
}
impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: LEARNING_RATE,
            l2_lambda: L2_LAMBDA,
            batch_size: BATCH_SZ,
            max_epochs: MAX_EPOCHS,
            early_stop_patience: EARLY_STOP_PATIENCE,
            validation_split: VALIDATION_SPLIT,
            rng_seed: None,
            k_fold_total: 0,
            k_fold_index: 0,
            data_augmentation_std: 0.0,
        }
    }
}

fn build_windows(embs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if embs.len() < WINDOW_SIZE {
        return vec![];
    }
    (0..=(embs.len() - WINDOW_SIZE))
        .map(|i| {
            let mut w = Vec::with_capacity(INPUT_DIM);
            for j in 0..WINDOW_SIZE {
                w.extend_from_slice(&embs[i + j]);
            }
            w
        })
        .collect()
}

struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
}
impl AdamState {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        }
    }
    fn update(&mut self, p: &mut [f32], g: &[f32], lr: f32) {
        self.t += 1;
        let b1 = ADAM_BETA1;
        let b2 = ADAM_BETA2;
        let lr_t = lr * (1.0 - b2.powi(self.t as i32)).sqrt() / (1.0 - b1.powi(self.t as i32));
        for i in 0..p.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g[i] * g[i];
            p[i] -= lr_t * self.m[i] / (self.v[i].sqrt() + ADAM_EPS);
        }
    }
}

/// Train the classifier using pure-Rust backprop + Adam.
pub fn train_classifier(
    pos: &[Vec<f32>],
    neg: &[Vec<f32>],
    cfg: &TrainingConfig,
) -> Result<ClassifierTrainingResult> {
    let pos_w = build_windows(pos);
    let neg_w = build_windows(neg);
    anyhow::ensure!(pos_w.len() + neg_w.len() >= 2, "Need ≥2 training windows");

    // Class-balanced weights
    let np = pos_w.len() as f32;
    let nn = neg_w.len() as f32;
    let total = np + nn;
    let pw = if np > 0.0 { total / (2.0 * np) } else { 0.0 };
    let nw = if nn > 0.0 { total / (2.0 * nn) } else { 0.0 };

    let mut all_x = Vec::with_capacity(pos_w.len() + neg_w.len());
    let mut all_y = Vec::with_capacity(pos_w.len() + neg_w.len());
    let mut all_w = Vec::with_capacity(pos_w.len() + neg_w.len());
    for w in &pos_w {
        all_x.push(w.clone());
        all_y.push(1.0);
        all_w.push(pw);
    }
    for w in &neg_w {
        all_x.push(w.clone());
        all_y.push(0.0);
        all_w.push(nw);
    }

    // L2-normalize
    for x in &mut all_x {
        let n = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
        for v in x {
            *v /= n;
        }
    }

    // Train/val split — stratified k-fold (mahbot-847) or random 80/20 fallback.
    let (tr_x, tr_y, tr_w, va_x, va_y, va_w, n_tr, _n_val, mut rng) = if cfg.k_fold_total > 0 {
        // Stratified k-fold: each class is partitioned separately to
        // preserve class ratios in every fold.
        let k = cfg.k_fold_total;
        let fold = cfg.k_fold_index.min(k - 1);
        let pos_count = pos_w.len();
        let neg_count = neg_w.len();

        let mut rng_r: Box<dyn rand::Rng> = if let Some(seed) = cfg.rng_seed {
            Box::new(rand::rngs::StdRng::seed_from_u64(seed))
        } else {
            Box::new(rand::rng())
        };

        // Shuffle class-specific indices
        let mut pos_idx: Vec<usize> = (0..pos_count).collect();
        pos_idx.shuffle(&mut rng_r);
        let mut neg_idx: Vec<usize> = (0..neg_count).collect();
        neg_idx.shuffle(&mut rng_r);

        let fold_for = |shuffled_pos: usize, class_size: usize, total_k: usize| -> usize {
            if class_size == 0 {
                return 0;
            }
            (shuffled_pos * total_k) / class_size
        };

        let mut tr_x = Vec::new();
        let mut tr_y = Vec::new();
        let mut tr_w = Vec::new();
        let mut va_x = Vec::new();
        let mut va_y = Vec::new();
        let mut va_w = Vec::new();

        // Positives: use shuffled index (_global_i) so each fold gets a
        // random subset rather than contiguous blocks.
        for (local_i, &global_i) in pos_idx.iter().enumerate() {
            let f = fold_for(local_i, pos_count, k);
            if f == fold {
                va_x.push(all_x[global_i].clone());
                va_y.push(all_y[global_i]);
                va_w.push(all_w[global_i]);
            } else {
                tr_x.push(all_x[global_i].clone());
                tr_y.push(all_y[global_i]);
                tr_w.push(all_w[global_i]);
            }
        }

        // Negatives: use shuffled index from neg_idx.
        for (local_i, &neg_gi) in neg_idx.iter().enumerate() {
            let global_i = pos_count + neg_gi;
            let f = fold_for(local_i, neg_count, k);
            if f == fold {
                va_x.push(all_x[global_i].clone());
                va_y.push(all_y[global_i]);
                va_w.push(all_w[global_i]);
            } else {
                tr_x.push(all_x[global_i].clone());
                tr_y.push(all_y[global_i]);
                tr_w.push(all_w[global_i]);
            }
        }

        let n_tr = tr_x.len();
        let n_val = va_x.len();
        info!(
            "Training (k-fold fold={fold}/{k}): {n_tr} train + {n_val} val \
                 ({pos_count} pos + {neg_count} neg windows)"
        );
        (tr_x, tr_y, tr_w, va_x, va_y, va_w, n_tr, n_val, rng_r)
    } else {
        // Original random 80/20 split
        let n = all_x.len();
        let n_val_calc = ((n as f32) * cfg.validation_split).ceil() as usize;
        let n_val_calc = n_val_calc.max(1).min(n - 1);
        let n_tr_calc = n - n_val_calc;

        let mut rng_r: Box<dyn rand::Rng> = if let Some(seed) = cfg.rng_seed {
            Box::new(rand::rngs::StdRng::seed_from_u64(seed))
        } else {
            Box::new(rand::rng())
        };
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(&mut rng_r);

        let tr_x = gather(&all_x, &idx[n_val_calc..]);
        let tr_y = gather(&all_y, &idx[n_val_calc..]);
        let tr_w = gather(&all_w, &idx[n_val_calc..]);
        let va_x = gather(&all_x, &idx[..n_val_calc]);
        let va_y = gather(&all_y, &idx[..n_val_calc]);
        let va_w = gather(&all_w, &idx[..n_val_calc]);

        info!(
            "Training (random split): {n_tr_calc} train + {n_val_calc} val \
                 ({np} pos + {nn} neg total)"
        );
        (
            tr_x, tr_y, tr_w, va_x, va_y, va_w, n_tr_calc, n_val_calc, rng_r,
        )
    };

    let cin = EMBEDDING_DIM;
    let lin = WINDOW_SIZE;
    let mut weights = ClassifierWeights::from_rng(&mut *rng);
    let bs = cfg.batch_size.min(n_tr).max(1);
    let mut best_loss = f32::INFINITY;
    let mut patience = 0;
    let mut best = weights.clone();

    let mut opt = AdamStateGroup::new(&weights);
    let mut epochs_trained = 0;

    for epoch in 0..cfg.max_epochs {
        epochs_trained = epoch + 1;
        let mut tr_idx: Vec<usize> = (0..n_tr).collect();
        tr_idx.shuffle(&mut rng);
        let lr_scale =
            0.5 * (1.0 + (std::f32::consts::PI * epoch as f32 / cfg.max_epochs as f32).cos());
        let lr = cfg.learning_rate * (0.001 + 0.999 * lr_scale);
        let mut epoch_loss = 0.0;
        let mut n_batches = 0;

        for chunk in tr_idx.chunks(bs) {
            let mut g = GradientBuffer::new(&weights);
            let mut batch_loss = 0.0;
            // BN running statistics accumulators (updated once per batch
            // instead of per-sample, which would otherwise compound the
            // momentum decay ~32× per batch, making the effective retention
            // 0.9³² ≈ 0.034 after a single batch).
            let mut bn1_mean_acc = vec![0.0; CONV1_OUT];
            let mut bn1_var_acc = vec![0.0; CONV1_OUT];
            let mut bn2_mean_acc = vec![0.0; CONV2_OUT];
            let mut bn2_var_acc = vec![0.0; CONV2_OUT];

            for (sample_idx, &i) in chunk.iter().enumerate() {
                let dropout_seed = cfg.rng_seed.unwrap_or_else(|| rand::rng().random::<u64>())
                    ^ (epoch as u64).wrapping_mul(1_000_000)
                    ^ sample_idx as u64;
                let mut ctx = TrainingCtx::new(dropout_seed);

                // Data augmentation (mahbot-847): add Gaussian noise and
                // re-normalize.  Uses the same per-sample RNG as dropout
                // for deterministic reproducibility.
                let x_cf = if cfg.data_augmentation_std > 0.0 {
                    let augmented =
                        apply_augmentation(&tr_x[i], cfg.data_augmentation_std, &mut ctx.rng);
                    to_channels_first(&augmented, cin, lin)
                } else {
                    to_channels_first(&tr_x[i], cin, lin)
                };
                let target = tr_y[i];
                let sw = tr_w[i];
                let pred = forward_pass_train(&x_cf, &weights, &mut ctx);
                let eps = 1e-7;
                let loss =
                    -sw * (target * (pred + eps).ln() + (1.0 - target) * (1.0 - pred + eps).ln());
                batch_loss += loss;
                backward(&x_cf, target, &weights, &mut g, Some(&ctx));

                // Accumulate per-sample BN stats for batch update.
                for (acc, &v) in bn1_mean_acc.iter_mut().zip(ctx.bn1_mean.iter()) {
                    *acc += v;
                }
                for (acc, &v) in bn1_var_acc.iter_mut().zip(ctx.bn1_var.iter()) {
                    *acc += v;
                }
                for (acc, &v) in bn2_mean_acc.iter_mut().zip(ctx.bn2_mean.iter()) {
                    *acc += v;
                }
                for (acc, &v) in bn2_var_acc.iter_mut().zip(ctx.bn2_var.iter()) {
                    *acc += v;
                }
            }
            // Update BN running stats once per batch using accumulated stats.
            let n_batch = chunk.len() as f32;
            for ci in 0..CONV1_OUT {
                let batch_mean = bn1_mean_acc[ci] / n_batch;
                let batch_var = bn1_var_acc[ci] / n_batch;
                weights.bn1_running_mean[ci] =
                    BN_MOMENTUM * weights.bn1_running_mean[ci] + (1.0 - BN_MOMENTUM) * batch_mean;
                weights.bn1_running_var[ci] =
                    BN_MOMENTUM * weights.bn1_running_var[ci] + (1.0 - BN_MOMENTUM) * batch_var;
            }
            for ci in 0..CONV2_OUT {
                let batch_mean = bn2_mean_acc[ci] / n_batch;
                let batch_var = bn2_var_acc[ci] / n_batch;
                weights.bn2_running_mean[ci] =
                    BN_MOMENTUM * weights.bn2_running_mean[ci] + (1.0 - BN_MOMENTUM) * batch_mean;
                weights.bn2_running_var[ci] =
                    BN_MOMENTUM * weights.bn2_running_var[ci] + (1.0 - BN_MOMENTUM) * batch_var;
            }
            // Average gradients
            let nf = chunk.len() as f32;
            for gv in g.all_mut() {
                for v in gv {
                    *v /= nf;
                }
            }
            // Gradient clipping (mahbot-846): cap global L2 norm to prevent
            // a single unlucky mini-batch from derailing optimization.
            {
                let mut sq_sum = 0.0;
                for gv in g.all_mut() {
                    for &v in gv.iter() {
                        sq_sum += v * v;
                    }
                }
                let norm = sq_sum.sqrt();
                if norm > GRADIENT_CLIP_NORM {
                    let scale = GRADIENT_CLIP_NORM / norm;
                    for gv in g.all_mut() {
                        for v in gv {
                            *v *= scale;
                        }
                    }
                }
            }
            // L2 regularization (applied to gradients before Adam, not
            // decoupled weight decay / AdamW).  This means the regularization
            // strength is modulated by Adam's adaptive learning rates per
            // parameter — intentional choice for simplicity, consistent with
            // the non-decoupled pattern used in many embedded MLP systems.
            // For decoupled weight decay (Loshchilov & Hutter 2019), switch
            // to subtracting `lr * l2 * param` directly in the update step.
            let l2 = cfg.l2_lambda;
            for (gv, wv) in g.conv1_w.iter_mut().zip(weights.conv1_weight.iter()) {
                *gv += l2 * wv;
            }
            for (gv, wv) in g.conv2_w.iter_mut().zip(weights.conv2_weight.iter()) {
                *gv += l2 * wv;
            }
            for (gv, wv) in g.fc_w.iter_mut().zip(weights.fc_weight.iter()) {
                *gv += l2 * wv;
            }
            // Adam step
            opt.step(&mut weights, &g, lr);
            epoch_loss += batch_loss / nf;
            n_batches += 1;
        }

        let val_loss = if va_x.is_empty() {
            f32::INFINITY
        } else {
            let mut vl = 0.0;
            for i in 0..va_x.len() {
                let x_cf = to_channels_first(&va_x[i], cin, lin);
                let pred = forward_pass_infer(&x_cf, &weights);
                let eps = 1e-7;
                let l = -va_w[i]
                    * (va_y[i] * (pred + eps).ln() + (1.0 - va_y[i]) * (1.0 - pred + eps).ln());
                vl += l;
            }
            vl / va_x.len() as f32
                + 0.5
                    * cfg.l2_lambda
                    * (weights.conv1_weight.iter().map(|x| x * x).sum::<f32>()
                        + weights.conv2_weight.iter().map(|x| x * x).sum::<f32>()
                        + weights.fc_weight.iter().map(|x| x * x).sum::<f32>())
        };

        info!(
            "Epoch {}/{}: loss={:.6} val={:.6}",
            epoch + 1,
            cfg.max_epochs,
            epoch_loss / n_batches as f32,
            val_loss
        );
        if val_loss < best_loss - 1e-6 {
            best_loss = val_loss;
            patience = 0;
            best = weights.clone();
        } else {
            patience += 1;
            if patience >= cfg.early_stop_patience {
                info!(
                    "Early stop at epoch {} (best val={:.6})",
                    epoch + 1,
                    best_loss
                );
                weights = best;
                break;
            }
        }
    }

    // Log average scores on ALL training positives and negatives.
    // We re-normalize pos_w/neg_w the same way all_x was normalized above.
    let l2_normalize = |windows: &[Vec<f32>]| -> Vec<Vec<f32>> {
        windows
            .iter()
            .map(|w| {
                let n = w.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
                w.iter().map(|v| v / n).collect()
            })
            .collect()
    };
    let score_normalized = |windows: &[Vec<f32>]| -> Vec<f32> {
        let norm = l2_normalize(windows);
        norm.iter()
            .map(|w| {
                let x_cf = to_channels_first(w, cin, lin);
                forward_pass_infer(&x_cf, &weights)
            })
            .collect()
    };
    let pos_scores = score_normalized(&pos_w);
    let neg_scores = score_normalized(&neg_w);
    let (pos_scores_mean, pos_scores_min, pos_scores_max) = compute_score_stats(&pos_scores);
    let (neg_scores_mean, neg_scores_min, neg_scores_max) = compute_score_stats(&neg_scores);
    info!(
        "Classifier final pos scores: mean={pos_scores_mean:.4} min={pos_scores_min:.4} max={pos_scores_max:.4} \
         neg: mean={neg_scores_mean:.4} min={neg_scores_min:.4} max={neg_scores_max:.4}",
    );

    weights.validate()?;
    Ok(ClassifierTrainingResult {
        weights: vec![weights],
        epochs_trained,
        best_val_loss: best_loss,
        val_losses: vec![best_loss],
        pos_scores_mean,
        pos_scores_min,
        pos_scores_max,
        neg_scores_mean,
        neg_scores_min,
        neg_scores_max,
    })
}

fn compute_score_stats(scores: &[f32]) -> (f32, f32, f32) {
    if scores.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mean = scores.iter().copied().sum::<f32>() / scores.len() as f32;
    let min = scores.iter().copied().fold(f32::INFINITY, f32::min);
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (mean, min, max)
}

fn gather<T: Clone>(data: &[T], idx: &[usize]) -> Vec<T> {
    idx.iter().map(|&i| data[i].clone()).collect()
}

fn to_channels_first(x: &[f32], cin: usize, lin: usize) -> Vec<f32> {
    let mut out = vec![0.0; cin * lin];
    for (t, chunk) in x.chunks(cin).enumerate() {
        for (c, &v) in chunk.iter().enumerate() {
            if c < cin && t < lin {
                out[c * lin + t] = v;
            }
        }
    }
    out
}

// ── Manual backprop ─────────────────────────────────────────────────────
//
// NOTE: Adding a new weight tensor to `ClassifierWeights` requires updating
// all ~7 touch points below: GradientBuffer fields + `new` + `all_mut`,
// AdamStateGroup fields + `new` + `step`, L2 regularization loop in
// `train_classifier`, `validate()`, `Default for ClassifierWeights`, and
// `param_count` (used in tests).  Missing any one causes silent gradient
// omission.  This is inherent to manual backprop without autograd.

struct GradientBuffer {
    conv1_w: Vec<f32>,
    conv1_b: Vec<f32>,
    bn1_gamma: Vec<f32>,
    bn1_beta: Vec<f32>,
    conv2_w: Vec<f32>,
    conv2_b: Vec<f32>,
    bn2_gamma: Vec<f32>,
    bn2_beta: Vec<f32>,
    fc_w: Vec<f32>,
    fc_b: Vec<f32>,
}
impl GradientBuffer {
    fn new(w: &ClassifierWeights) -> Self {
        Self {
            conv1_w: vec![0.0; w.conv1_weight.len()],
            conv1_b: vec![0.0; w.conv1_bias.len()],
            bn1_gamma: vec![0.0; w.bn1_gamma.len()],
            bn1_beta: vec![0.0; w.bn1_beta.len()],
            conv2_w: vec![0.0; w.conv2_weight.len()],
            conv2_b: vec![0.0; w.conv2_bias.len()],
            bn2_gamma: vec![0.0; w.bn2_gamma.len()],
            bn2_beta: vec![0.0; w.bn2_beta.len()],
            fc_w: vec![0.0; w.fc_weight.len()],
            fc_b: vec![0.0; w.fc_bias.len()],
        }
    }
    fn all_mut(&mut self) -> Vec<&mut [f32]> {
        vec![
            &mut self.conv1_w,
            &mut self.conv1_b,
            &mut self.bn1_gamma,
            &mut self.bn1_beta,
            &mut self.conv2_w,
            &mut self.conv2_b,
            &mut self.bn2_gamma,
            &mut self.bn2_beta,
            &mut self.fc_w,
            &mut self.fc_b,
        ]
    }
}

struct AdamStateGroup {
    conv1_w: AdamState,
    conv1_b: AdamState,
    bn1_gamma: AdamState,
    bn1_beta: AdamState,
    conv2_w: AdamState,
    conv2_b: AdamState,
    bn2_gamma: AdamState,
    bn2_beta: AdamState,
    fc_w: AdamState,
    fc_b: AdamState,
}
impl AdamStateGroup {
    fn new(w: &ClassifierWeights) -> Self {
        Self {
            conv1_w: AdamState::new(w.conv1_weight.len()),
            conv1_b: AdamState::new(w.conv1_bias.len()),
            bn1_gamma: AdamState::new(w.bn1_gamma.len()),
            bn1_beta: AdamState::new(w.bn1_beta.len()),
            conv2_w: AdamState::new(w.conv2_weight.len()),
            conv2_b: AdamState::new(w.conv2_bias.len()),
            bn2_gamma: AdamState::new(w.bn2_gamma.len()),
            bn2_beta: AdamState::new(w.bn2_beta.len()),
            fc_w: AdamState::new(w.fc_weight.len()),
            fc_b: AdamState::new(w.fc_bias.len()),
        }
    }
    fn step(&mut self, w: &mut ClassifierWeights, g: &GradientBuffer, lr: f32) {
        self.conv1_w.update(&mut w.conv1_weight, &g.conv1_w, lr);
        self.conv1_b.update(&mut w.conv1_bias, &g.conv1_b, lr);
        self.bn1_gamma.update(&mut w.bn1_gamma, &g.bn1_gamma, lr);
        self.bn1_beta.update(&mut w.bn1_beta, &g.bn1_beta, lr);
        self.conv2_w.update(&mut w.conv2_weight, &g.conv2_w, lr);
        self.conv2_b.update(&mut w.conv2_bias, &g.conv2_b, lr);
        self.bn2_gamma.update(&mut w.bn2_gamma, &g.bn2_gamma, lr);
        self.bn2_beta.update(&mut w.bn2_beta, &g.bn2_beta, lr);
        self.fc_w.update(&mut w.fc_weight, &g.fc_w, lr);
        self.fc_b.update(&mut w.fc_bias, &g.fc_b, lr);
    }
}

/// Manual backward pass. Accumulates gradients into `g`.
///
/// When `ctx` is `Some` (training mode, mahbot-846):
/// - Batch-normalisation gradients use the per-sample batch statistics
///   recorded by `forward_pass_train` instead of running statistics.
/// - Dropout masks from the forward pass are applied to the ReLU gradients.
///
/// When `ctx` is `None` (legacy/inference backward):
/// - Uses running statistics (identical to original `backward`).
#[allow(clippy::cast_precision_loss)]
fn backward(
    x: &[f32],
    target: f32,
    w: &ClassifierWeights,
    g: &mut GradientBuffer,
    ctx: Option<&TrainingCtx>,
) {
    let cin = EMBEDDING_DIM;
    let lin = WINDOW_SIZE;
    let c1 = CONV1_OUT;
    let c2 = CONV2_OUT;
    let eps = w.bn_eps;

    // Determine which BN statistics to use.
    let (bn1_mean, bn1_var, bn2_mean, bn2_var) = match ctx {
        Some(ctx) => (&*ctx.bn1_mean, &*ctx.bn1_var, &*ctx.bn2_mean, &*ctx.bn2_var),
        None => (
            &*w.bn1_running_mean,
            &*w.bn1_running_var,
            &*w.bn2_running_mean,
            &*w.bn2_running_var,
        ),
    };

    // Forward intermediates
    let mut conv1_pre = vec![0.0; c1 * lin];
    for co in 0..c1 {
        for li in 0..lin {
            let mut s = w.conv1_bias[co];
            for ci in 0..cin {
                for k in 0..KERNEL_SIZE {
                    let ii = li as isize + k as isize - PADDING as isize;
                    if ii >= 0 && ii < lin as isize {
                        s += x[ci * lin + ii as usize]
                            * w.conv1_weight[(co * cin + ci) * KERNEL_SIZE + k];
                    }
                }
            }
            conv1_pre[co * lin + li] = s;
        }
    }

    let mut bn1_out = vec![0.0; c1 * lin];
    let mut bn1_xhat = vec![0.0; c1 * lin];
    let mut bn1_std = vec![0.0; c1];
    for ci in 0..c1 {
        let std = (bn1_var[ci] + eps).sqrt();
        bn1_std[ci] = std;
        for li in 0..lin {
            let idx = ci * lin + li;
            bn1_xhat[idx] = (conv1_pre[idx] - bn1_mean[ci]) / std;
            bn1_out[idx] = w.bn1_gamma[ci] * bn1_xhat[idx] + w.bn1_beta[ci];
        }
    }

    let mut relu1 = vec![0.0; c1 * lin];
    let mut relu1m = vec![0.0; c1 * lin];
    for i in 0..(c1 * lin) {
        relu1[i] = bn1_out[i].max(0.0);
        relu1m[i] = if bn1_out[i] > 0.0 { 1.0 } else { 0.0 };
    }

    // Apply dropout mask1 (training) or identity (inference)
    if let Some(ctx) = ctx {
        for i in 0..(c1 * lin) {
            relu1[i] *= ctx.dropout_mask1[i];
            // Mask gradient through ReLU: the dropout mask is 0.0 for
            // dropped units, so the gradient will also be zeroed.
            relu1m[i] *= ctx.dropout_mask1[i];
        }
    }

    let mut conv2_pre = vec![0.0; c2 * lin];
    for co in 0..c2 {
        for li in 0..lin {
            let mut s = w.conv2_bias[co];
            for ci in 0..c1 {
                for k in 0..KERNEL_SIZE {
                    let ii = li as isize + k as isize - PADDING as isize;
                    if ii >= 0 && ii < lin as isize {
                        s += relu1[ci * lin + ii as usize]
                            * w.conv2_weight[(co * c1 + ci) * KERNEL_SIZE + k];
                    }
                }
            }
            conv2_pre[co * lin + li] = s;
        }
    }

    let mut bn2_out = vec![0.0; c2 * lin];
    let mut bn2_xhat = vec![0.0; c2 * lin];
    let mut bn2_std = vec![0.0; c2];
    for ci in 0..c2 {
        let std = (bn2_var[ci] + eps).sqrt();
        bn2_std[ci] = std;
        for li in 0..lin {
            let idx = ci * lin + li;
            bn2_xhat[idx] = (conv2_pre[idx] - bn2_mean[ci]) / std;
            bn2_out[idx] = w.bn2_gamma[ci] * bn2_xhat[idx] + w.bn2_beta[ci];
        }
    }

    let mut relu2 = vec![0.0; c2 * lin];
    let mut relu2m = vec![0.0; c2 * lin];
    for i in 0..(c2 * lin) {
        relu2[i] = bn2_out[i].max(0.0);
        relu2m[i] = if bn2_out[i] > 0.0 { 1.0 } else { 0.0 };
    }

    // Apply dropout mask2 (training) or identity (inference)
    if let Some(ctx) = ctx {
        for i in 0..(c2 * lin) {
            relu2[i] *= ctx.dropout_mask2[i];
            relu2m[i] *= ctx.dropout_mask2[i];
        }
    }

    let mut pooled = vec![0.0; c2];
    for ci in 0..c2 {
        let mut s = 0.0;
        for li in 0..lin {
            s += relu2[ci * lin + li];
        }
        pooled[ci] = s / lin as f32;
    }

    let logit = dot(&pooled, &w.fc_weight) + w.fc_bias[0];
    let pred = sigmoid(logit);
    let d_logit = pred - target;

    // FC grads
    for j in 0..c2 {
        g.fc_w[j] += pooled[j] * d_logit;
    }
    g.fc_b[0] += d_logit;

    let mut d_pooled = vec![0.0; c2];
    for j in 0..c2 {
        d_pooled[j] = w.fc_weight[j] * d_logit;
    }

    let mut d_relu2 = vec![0.0; c2 * lin];
    for ci in 0..c2 {
        let grad = d_pooled[ci] / lin as f32;
        for li in 0..lin {
            d_relu2[ci * lin + li] = grad;
        }
    }

    let mut d_bn2 = vec![0.0; c2 * lin];
    for i in 0..(c2 * lin) {
        d_bn2[i] = d_relu2[i] * relu2m[i];
    }

    let mut d_conv2 = vec![0.0; c2 * lin];
    for ci in 0..c2 {
        let inv_std = 1.0 / bn2_std[ci];
        for li in 0..lin {
            let idx = ci * lin + li;
            d_conv2[idx] = d_bn2[idx] * w.bn2_gamma[ci] * inv_std;
        }
        let mut dg = 0.0;
        let mut db = 0.0;
        for li in 0..lin {
            dg += d_bn2[ci * lin + li] * bn2_xhat[ci * lin + li];
            db += d_bn2[ci * lin + li];
        }
        g.bn2_gamma[ci] += dg;
        g.bn2_beta[ci] += db;
    }

    // Conv2 backward
    let mut d_relu1 = vec![0.0; c1 * lin];
    for co in 0..c2 {
        for li in 0..lin {
            let go = d_conv2[co * lin + li];
            for ci in 0..c1 {
                for k in 0..KERNEL_SIZE {
                    let ii = li as isize + k as isize - PADDING as isize;
                    if ii >= 0 && ii < lin as isize {
                        let widx = (co * c1 + ci) * KERNEL_SIZE + k;
                        g.conv2_w[widx] += go * relu1[ci * lin + ii as usize];
                        d_relu1[ci * lin + ii as usize] += go * w.conv2_weight[widx];
                    }
                }
            }
            g.conv2_b[co] += go;
        }
    }

    let mut d_bn1 = vec![0.0; c1 * lin];
    for i in 0..(c1 * lin) {
        d_bn1[i] = d_relu1[i] * relu1m[i];
    }

    let mut d_conv1 = vec![0.0; c1 * lin];
    for ci in 0..c1 {
        let inv_std = 1.0 / bn1_std[ci];
        for li in 0..lin {
            let idx = ci * lin + li;
            d_conv1[idx] = d_bn1[idx] * w.bn1_gamma[ci] * inv_std;
        }
        let mut dg = 0.0;
        let mut db = 0.0;
        for li in 0..lin {
            dg += d_bn1[ci * lin + li] * bn1_xhat[ci * lin + li];
            db += d_bn1[ci * lin + li];
        }
        g.bn1_gamma[ci] += dg;
        g.bn1_beta[ci] += db;
    }

    // Conv1 backward
    for co in 0..c1 {
        for li in 0..lin {
            let go = d_conv1[co * lin + li];
            for ci in 0..cin {
                for k in 0..KERNEL_SIZE {
                    let ii = li as isize + k as isize - PADDING as isize;
                    if ii >= 0 && ii < lin as isize {
                        let widx = (co * cin + ci) * KERNEL_SIZE + k;
                        g.conv1_w[widx] += go * x[ci * lin + ii as usize];
                    }
                }
            }
            g.conv1_b[co] += go;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_pos() {
        let s = sigmoid(5.0);
        assert!(s > 0.99);
    }
    #[test]
    fn test_sigmoid_neg() {
        let s = sigmoid(-5.0);
        assert!(s < 0.01);
    }
    #[test]
    fn test_sigmoid_zero() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_forward_constant() {
        let embs: Vec<Vec<f32>> = (0..WINDOW_SIZE).map(|_| vec![0.5; EMBEDDING_DIM]).collect();
        let w = ClassifierWeights {
            conv1_weight: vec![0.0; CONV1_OUT * EMBEDDING_DIM * KERNEL_SIZE],
            conv1_bias: vec![0.0; CONV1_OUT],
            bn1_gamma: vec![1.0; CONV1_OUT],
            bn1_beta: vec![0.0; CONV1_OUT],
            bn1_running_mean: vec![0.0; CONV1_OUT],
            bn1_running_var: vec![1.0; CONV1_OUT],
            conv2_weight: vec![0.0; CONV2_OUT * CONV1_OUT * KERNEL_SIZE],
            conv2_bias: vec![0.0; CONV2_OUT],
            bn2_gamma: vec![1.0; CONV2_OUT],
            bn2_beta: vec![0.0; CONV2_OUT],
            bn2_running_mean: vec![0.0; CONV2_OUT],
            bn2_running_var: vec![1.0; CONV2_OUT],
            fc_weight: vec![0.0; CONV2_OUT * FC_OUT],
            fc_bias: vec![0.0; FC_OUT],
            bn_eps: 1e-5,
        };
        let c = WakeWordClassifier::new(w);
        let score = c.forward(&embs);
        assert!((score - 0.5).abs() < 1e-4, "Expected 0.5, got {score}");
    }

    #[test]
    fn test_build_windows_basic() {
        let embs: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32; EMBEDDING_DIM]).collect();
        assert_eq!(build_windows(&embs).len(), 3);
    }
    #[test]
    fn test_build_windows_empty() {
        assert!(build_windows(&[]).is_empty());
    }
    #[test]
    fn test_build_windows_short() {
        let embs: Vec<Vec<f32>> = (0..2).map(|i| vec![i as f32; EMBEDDING_DIM]).collect();
        assert!(build_windows(&embs).is_empty());
    }

    #[test]
    fn test_weights_serde() {
        let w = ClassifierWeights::default();
        let json = serde_json::to_string(&w).unwrap();
        let _: ClassifierWeights = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_weights_serde_legacy() {
        // Legacy JSON without BN running stats fields — these must be absent
        // so that #[serde(default)] kicks in, verifying backward compatibility
        // with enrollment data serialized before mahbot-846 added the four
        // BN running-stat fields.
        let legacy = serde_json::json!({
            "conv1_weight": vec![0.0; CONV1_OUT * EMBEDDING_DIM * KERNEL_SIZE],
            "conv1_bias": vec![0.0; CONV1_OUT],
            "bn1_gamma": vec![1.0; CONV1_OUT],
            "bn1_beta": vec![0.0; CONV1_OUT],
            "conv2_weight": vec![0.0; CONV2_OUT * CONV1_OUT * KERNEL_SIZE],
            "conv2_bias": vec![0.0; CONV2_OUT],
            "bn2_gamma": vec![1.0; CONV2_OUT],
            "bn2_beta": vec![0.0; CONV2_OUT],
            "fc_weight": vec![0.0; CONV2_OUT * FC_OUT],
            "fc_bias": vec![0.0; FC_OUT],
            "bn_eps": 1e-5_f32,
        });
        let w: ClassifierWeights = serde_json::from_value(legacy).unwrap();
        // Verify BN running stats were defaulted to correct shapes and values.
        assert_eq!(w.bn1_running_mean.len(), CONV1_OUT);
        assert_eq!(w.bn1_running_var.len(), CONV1_OUT);
        assert_eq!(w.bn2_running_mean.len(), CONV2_OUT);
        assert_eq!(w.bn2_running_var.len(), CONV2_OUT);
        assert!(w.bn1_running_mean.iter().all(|&v| v == 0.0));
        assert!(w.bn1_running_var.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(w.bn2_running_mean.iter().all(|&v| v == 0.0));
        assert!(w.bn2_running_var.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        // Also verify the rest round-trips correctly.
        assert!(w.validate().is_ok());
    }

    #[test]
    fn test_validate_fails() {
        let mut w = ClassifierWeights::default();
        w.conv1_weight.push(0.0);
        assert!(w.validate().is_err());
    }

    #[test]
    fn test_validate_passes() {
        let w = ClassifierWeights::default();
        assert!(w.validate().is_ok());
    }

    #[test]
    fn test_relu() {
        let mut x = vec![-1.0, 0.0, 2.0, -0.5, 3.0];
        relu(&mut x);
        assert_eq!(x, vec![0.0, 0.0, 2.0, 0.0, 3.0]);
    }

    #[test]
    fn test_dot() {
        assert!((dot(&[1.0, 2.0], &[3.0, 4.0]) - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_adaptive_avg_pool() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let p = adaptive_avg_pool(&x, 2, 3);
        assert!((p[0] - 2.0).abs() < 1e-6);
        assert!((p[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_train_classifier_convergence() {
        // Generate two separable clusters in embedding space.
        // Positive cluster centered at +0.3, negative at -0.3, with noise.
        // Each embedding is EMBEDDING_DIM-length; build_windows groups
        // WINDOW_SIZE embeddings into each training window — matching the
        // production data pipeline in voice.rs.
        let mut rng = rand::rng();
        let mut make_emb = |center: f32, noise: f32| -> Vec<f32> {
            (0..EMBEDDING_DIM)
                .map(|_| center + (rng.random::<f32>() - 0.5) * noise)
                .collect()
        };

        // 100 windows each = 300 embeddings (WINDOW_SIZE per window).
        let n_wins = 100;
        let n_embs = n_wins * WINDOW_SIZE;
        let pos: Vec<Vec<f32>> = (0..n_embs).map(|_| make_emb(0.3, 0.4)).collect();
        let neg: Vec<Vec<f32>> = (0..n_embs).map(|_| make_emb(-0.3, 0.4)).collect();

        let cfg = TrainingConfig {
            max_epochs: 50,
            ..Default::default()
        };
        let result = train_classifier(&pos, &neg, &cfg).unwrap();
        let classifier = WakeWordClassifier::new_ensemble(result.weights);
        // Evaluate on the windows produced by build_windows — same path
        // that train_classifier uses internally.
        for win in build_windows(&pos) {
            let embs: Vec<Vec<f32>> = (0..WINDOW_SIZE)
                .map(|t| {
                    let start = t * EMBEDDING_DIM;
                    win[start..start + EMBEDDING_DIM].to_vec()
                })
                .collect();
            let score = classifier.forward(&embs);
            assert!(
                score > 0.8,
                "Positive window should score >0.8, got {score}"
            );
        }
        for win in build_windows(&neg) {
            let embs: Vec<Vec<f32>> = (0..WINDOW_SIZE)
                .map(|t| {
                    let start = t * EMBEDDING_DIM;
                    win[start..start + EMBEDDING_DIM].to_vec()
                })
                .collect();
            let score = classifier.forward(&embs);
            assert!(
                score < 0.2,
                "Negative window should score <0.2, got {score}"
            );
        }
    }

    #[test]
    fn test_apply_augmentation_preserves_unit_norm() {
        // Build a random unit-normalized vector.
        let mut rng = rand::rng();
        let mut x = vec![0.0; EMBEDDING_DIM];
        for v in &mut x {
            *v = rng.random::<f32>() - 0.5;
        }
        let n = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
        for v in &mut x {
            *v /= n;
        }
        let orig_norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((orig_norm - 1.0).abs() < 1e-6);

        let augmented = apply_augmentation(&x, 0.05, &mut rng);
        let aug_norm = augmented.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (aug_norm - 1.0).abs() < 1e-4,
            "Augmented vector must remain unit-normalized, got norm={aug_norm}",
        );
    }

    #[test]
    fn test_apply_augmentation_produces_variation() {
        // Two augmentations of the same vector should differ from the
        // original and from each other.
        let mut rng = rand::rng();
        let x = vec![1.0 / (EMBEDDING_DIM as f32).sqrt(); EMBEDDING_DIM];
        let a1 = apply_augmentation(&x, 0.05, &mut rng);
        let a2 = apply_augmentation(&x, 0.05, &mut rng);
        // Cosine similarity with original should be < 1.0
        let dot1: f32 = x.iter().zip(a1.iter()).map(|(a, b)| a * b).sum();
        assert!(
            dot1 < 0.99,
            "Augmented vector should differ from original (cos sim={dot1})",
        );
        // Two augmentations should differ from each other
        let dot12: f32 = a1.iter().zip(a2.iter()).map(|(a, b)| a * b).sum();
        assert!(
            dot12 < 0.99,
            "Two augmented vectors should differ (cos sim={dot12})",
        );
    }

    #[test]
    fn test_compute_softmax_weights_basic() {
        let losses = vec![1.0, 2.0, 3.0];
        let weights = WakeWordClassifier::compute_softmax_weights(&losses, 3);
        assert_eq!(weights.len(), 3);
        // Best model (lowest loss) gets highest weight
        assert!(
            weights[0] > weights[1] && weights[1] > weights[2],
            "Weights should be strictly decreasing with increasing loss: {weights:?}",
        );
        // Weights sum to 1.0
        let wsum: f32 = weights.iter().sum();
        assert!(
            (wsum - 1.0).abs() < 1e-6,
            "Weights should sum to 1.0, got {wsum}",
        );
    }

    #[test]
    fn test_ensemble_weighted_length_mismatch() {
        // When val_losses length doesn't match member count,
        // new_ensemble_weighted falls back to uniform averaging.
        let mut rng = rand::rng();
        let mut make_weights = || -> ClassifierWeights {
            let mut w = ClassifierWeights::default();
            w.fc_bias[0] = rng.random::<f32>() * 2.0 - 1.0;
            w
        };
        let members = vec![make_weights(), make_weights(), make_weights()];
        let losses = vec![1.0, 2.0]; // 2 losses for 3 members
        let classifier = WakeWordClassifier::new_ensemble_weighted(members, losses);
        // Cached weights should be uniform (1/3).
        assert_eq!(classifier.cached_weights.len(), 3);
        let expected = 1.0 / 3.0;
        for &w in &classifier.cached_weights {
            assert!(
                (w - expected).abs() < 1e-6,
                "Uniform fallback expected {expected}, got {w}",
            );
        }
        assert!(
            classifier.member_val_losses.is_empty(),
            "Should store no val_losses on length mismatch",
        );
    }

    #[test]
    fn test_compute_softmax_weights_nan_guard() {
        // NaN in val_losses should produce uniform fallback.
        let losses = vec![1.0, f32::NAN, 3.0];
        let weights = WakeWordClassifier::compute_softmax_weights(&losses, 3);
        assert_eq!(weights.len(), 3);
        let expected = 1.0 / 3.0;
        for &w in &weights {
            assert!(
                (w - expected).abs() < 1e-6,
                "NaN guard should fall back to uniform, got {w}",
            );
        }
        // All weights must be finite (not NaN).
        assert!(
            weights.iter().all(|w| w.is_finite()),
            "All weights must be finite: {weights:?}",
        );
    }

    #[test]
    fn test_ensemble_weighted_vs_uniform() {
        // A 3-member ensemble where members produce different scores.
        // Weighted averaging should give different result from uniform.
        let mut rng = rand::rng();
        let make_weights = |bias: f32| -> ClassifierWeights {
            let mut w = ClassifierWeights::default();
            // Set fc_bias to produce different post-sigmoid scores.
            w.fc_bias[0] = bias;
            w
        };
        let members = vec![make_weights(0.5), make_weights(0.0), make_weights(-0.5)];
        let embs: Vec<Vec<f32>> = (0..WINDOW_SIZE).map(|_| vec![0.3; EMBEDDING_DIM]).collect();

        let uniform = WakeWordClassifier::new_ensemble(members.clone());
        let unweighted = uniform.forward(&embs);

        let weighted = WakeWordClassifier::new_ensemble_weighted(members, vec![1.0, 2.0, 3.0]);
        let weighted_score = weighted.forward(&embs);

        assert!(
            (weighted_score - unweighted).abs() > 0.001,
            "Weighted and uniform scores should differ: weighted={weighted_score} uniform={unweighted}",
        );
    }

    #[test]
    fn test_kfold_training_produces_valid_classifier() {
        // Generate synthetic positive and negative data, then verify that
        // k-fold training produces a valid classifier with correct fold sizes.
        let mut rng = rand::rng();
        let n_pos = 20;
        let n_neg = 40;
        let k_fold = 5;
        let fold_idx = 0;
        let pos: Vec<Vec<f32>> = (0..n_pos)
            .map(|_| {
                let mut v: Vec<f32> = (0..EMBEDDING_DIM)
                    .map(|_| rng.random::<f32>() * 0.2 + 0.4)
                    .collect();
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
                for x in &mut v {
                    *x /= n;
                }
                v
            })
            .collect();
        let neg: Vec<Vec<f32>> = (0..n_neg)
            .map(|_| {
                let mut v: Vec<f32> = (0..EMBEDDING_DIM)
                    .map(|_| rng.random::<f32>() * 0.2 - 0.4)
                    .collect();
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
                for x in &mut v {
                    *x /= n;
                }
                v
            })
            .collect();

        // TrainingConfig with k-fold = 5, fold index 0.
        // Fold index 0 uses 1/5 of each class for validation:
        // positive → 4 val (20/5), 16 train  (ratio ~0.20)
        // negative → 8 val (40/5), 32 train  (ratio ~0.20)
        // Each fold gets floor(N/k) = N/5 samples per class.
        let cfg = TrainingConfig {
            k_fold_total: k_fold,
            k_fold_index: fold_idx,
            ..Default::default()
        };
        let result = train_classifier(&pos, &neg, &cfg).unwrap();
        // Single member since we only pass one full training run.
        assert_eq!(result.weights.len(), 1);
        assert_eq!(
            result.val_losses.len(),
            1,
            "Should produce one val_loss per member",
        );

        // Verify the model converges despite having fewer training examples
        // (only 4/5 of the data).
        let classifier = WakeWordClassifier::new_ensemble(result.weights);
        for win in build_windows(&pos) {
            let embs: Vec<Vec<f32>> = (0..WINDOW_SIZE)
                .map(|t| {
                    let start = t * EMBEDDING_DIM;
                    win[start..start + EMBEDDING_DIM].to_vec()
                })
                .collect();
            assert!(
                classifier.forward(&embs) > 0.5,
                "k-fold trained classifier should score positive windows >0.5",
            );
        }
    }
}
