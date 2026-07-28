//! Conv1D wake word classifier — replaces DTW template matching.
//!
//! Architecture: Conv1D(96→64, k=3) + BN + ReLU → Conv1D(64→64, k=3) + BN + ReLU
//! → AdaptiveAvgPool1d → Linear(64→1) + Sigmoid.
//!
//! Batch-normalisation uses frozen identity statistics (mean=0, var=1)
//! acting as a learned per-channel affine transform — identical at train
//! and inference time.  This was found to be correct for a network where
//! per-sample normalization over only WINDOW_SIZE=3 spatial positions
//! destroys the magnitude differences learned by Conv1D layers (mahbot-849).
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

use crate::embedding_sequence::EmbeddingSequence;

use crate::EMBEDDING_DIM;
#[cfg(test)]
use crate::embedding_sequence::LabelStratum;

// ── Constants ────────────────────────────────────────────────────────────

pub const WINDOW_SIZE: usize = 3;

/// Number of ensemble members to train for wake word detection (mahbot-839).
/// Five independent models with different seeds are trained during enrollment
/// and their post-sigmoid scores are averaged at inference time.
pub const NUM_ENSEMBLE_MEMBERS: usize = 5;
pub const INPUT_DIM: usize = WINDOW_SIZE * EMBEDDING_DIM; // 288
/// Default baseline architecture values used for serialization defaults.
const DEFAULT_CONV1_OUT: usize = 64;
const DEFAULT_CONV2_OUT: usize = 64;
const DEFAULT_KERNEL_SIZE: usize = 3;
const FC_OUT: usize = 1;

/// Architecture configuration for a Conv1D ensemble member (mahbot-848).
///
/// Defines the Conv1D channel counts and kernel size that determine each
/// member's feature extraction capacity and temporal receptive field.
/// Different architectures across ensemble members produce diverse feature
/// representations that improve ensemble robustness on ambiguous inputs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ArchConfig {
    pub conv1_out: usize,
    pub conv2_out: usize,
    pub kernel_size: usize,
}

impl ArchConfig {
    /// Compute `padding` as `kernel_size / 2` for "same" convolution where
    /// the output spatial length equals the input length (odd kernel sizes
    /// only — 3, 5, …).
    #[must_use]
    pub const fn padding(&self) -> usize {
        self.kernel_size / 2
    }
}

impl Default for ArchConfig {
    fn default() -> Self {
        Self {
            conv1_out: DEFAULT_CONV1_OUT,
            conv2_out: DEFAULT_CONV2_OUT,
            kernel_size: DEFAULT_KERNEL_SIZE,
        }
    }
}

/// The five architecture variants used for ensemble feature diversity
/// (mahbot-848).  Each member learns genuinely different feature representations:
///
/// 1. Low capacity  (32/32/k3) — acts as a regularized baseline
/// 2. Baseline     (64/64/k3) — current default architecture
/// 3. Wide kernel  (64/64/k5) — wider temporal receptive field
/// 4. High channels (96/96/k3) — richer feature extraction
/// 5. High capacity (128/128/k3) — fine-grained temporal patterns
pub const ENSEMBLE_ARCHS: [ArchConfig; NUM_ENSEMBLE_MEMBERS] = [
    ArchConfig {
        conv1_out: 32,
        conv2_out: 32,
        kernel_size: 3,
    },
    ArchConfig {
        conv1_out: 64,
        conv2_out: 64,
        kernel_size: 3,
    },
    ArchConfig {
        conv1_out: 64,
        conv2_out: 64,
        kernel_size: 5,
    },
    ArchConfig {
        conv1_out: 96,
        conv2_out: 96,
        kernel_size: 3,
    },
    ArchConfig {
        conv1_out: 128,
        conv2_out: 128,
        kernel_size: 3,
    },
];

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

/// Maximum gradient norm for clipping (mahbot-846).  Set to 100.0
/// (effectively disabled) — on tiny datasets gradient clipping destroys
/// the weak learning signal (mahbot-851).  The value is high enough that
/// no realistic gradient will trigger it.
const GRADIENT_CLIP_NORM: f32 = 100.0;

// ── Weights ─────────────────────────────────────────────────────────────

/// Default running mean for BN1 (used when deserializing legacy enrollment
/// data that predates the BN training stats, preserving backward compatibility).
fn default_bn1_running_mean() -> Vec<f32> {
    vec![0.0; DEFAULT_CONV1_OUT]
}

/// Default running variance for BN1 (see [`default_bn1_running_mean`]).
fn default_bn1_running_var() -> Vec<f32> {
    vec![1.0; DEFAULT_CONV1_OUT]
}

/// Default running mean for BN2 (see [`default_bn1_running_mean`]).
fn default_bn2_running_mean() -> Vec<f32> {
    vec![0.0; DEFAULT_CONV2_OUT]
}

/// Default running variance for BN2 (see [`default_bn1_running_mean`]).
fn default_bn2_running_var() -> Vec<f32> {
    vec![1.0; DEFAULT_CONV2_OUT]
}

fn default_arch() -> ArchConfig {
    ArchConfig::default()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierWeights {
    pub conv1_weight: Vec<f32>, // [conv1_out, EMBEDDING_DIM, kernel_size]
    pub conv1_bias: Vec<f32>,   // [conv1_out]
    pub bn1_gamma: Vec<f32>,
    pub bn1_beta: Vec<f32>,
    #[serde(default = "default_bn1_running_mean")]
    pub bn1_running_mean: Vec<f32>,
    #[serde(default = "default_bn1_running_var")]
    pub bn1_running_var: Vec<f32>,
    pub conv2_weight: Vec<f32>, // [conv2_out, conv1_out, kernel_size]
    pub conv2_bias: Vec<f32>,   // [conv2_out]
    pub bn2_gamma: Vec<f32>,
    pub bn2_beta: Vec<f32>,
    #[serde(default = "default_bn2_running_mean")]
    pub bn2_running_mean: Vec<f32>,
    #[serde(default = "default_bn2_running_var")]
    pub bn2_running_var: Vec<f32>,
    pub fc_weight: Vec<f32>, // [1, conv2_out]
    pub fc_bias: Vec<f32>,   // [1]
    pub bn_eps: f32,
    /// Architecture configuration for this ensemble member (mahbot-848).
    /// Defines conv1_out, conv2_out, and kernel_size that were used during
    /// training.  Uses `#[serde(default)]` so legacy enrollment data (which
    /// predates this field) deserializes with the baseline architecture,
    /// preserving backward compatibility.
    #[serde(default = "default_arch")]
    pub arch: ArchConfig,
}

impl Default for ClassifierWeights {
    fn default() -> Self {
        Self::from_rng(&mut rand::rng(), &ArchConfig::default())
    }
}

impl ClassifierWeights {
    /// Initialize classifier weights using a seeded RNG for deterministic training
    /// and the given architecture configuration (mahbot-848).
    /// Used by [`train_classifier`] when a seed is configured, replacing the
    /// non-deterministic `Default` path.
    pub fn from_rng(rng: &mut (impl rand::Rng + ?Sized), arch: &ArchConfig) -> Self {
        let ks = arch.kernel_size;
        let c1 = arch.conv1_out;
        let c2 = arch.conv2_out;
        // Xavier/Glorot uniform initialization corrected for Conv1D
        // (mahbot-846).  fan_in and fan_out must include kernel_size.
        // Formula: scale = sqrt(6 / (fan_in + fan_out))
        //   fan_in = in_channels * kernel_size
        //   fan_out = out_channels * kernel_size
        // Multiplied by 1.7 (mahbot-851) — the "correct" Xavier scale
        // produces weights too small for this tiny dataset (~99 positive
        // windows) to escape the flat region of the loss landscape.
        // The oversized init was used pre-846 and is required for learning.
        let scale_c1 = 1.7 * (6.0 / ((EMBEDDING_DIM + c1) * ks) as f32).sqrt();
        let scale_c2 = 1.7 * (6.0 / ((c1 + c2) * ks) as f32).sqrt();
        let scale_fc = 1.7 * (6.0 / (c2 + FC_OUT) as f32).sqrt();
        let mut uniform =
            |s: f32, n: usize| -> Vec<f32> { (0..n).map(|_| rng.random_range(-s..s)).collect() };
        Self {
            conv1_weight: uniform(scale_c1, c1 * EMBEDDING_DIM * ks),
            conv1_bias: vec![0.0; c1],
            bn1_gamma: vec![1.0; c1],
            bn1_beta: vec![0.0; c1],
            bn1_running_mean: vec![0.0; c1],
            bn1_running_var: vec![1.0; c1],
            conv2_weight: uniform(scale_c2, c2 * c1 * ks),
            conv2_bias: vec![0.0; c2],
            bn2_gamma: vec![1.0; c2],
            bn2_beta: vec![0.0; c2],
            bn2_running_mean: vec![0.0; c2],
            bn2_running_var: vec![1.0; c2],
            fc_weight: uniform(scale_fc, c2 * FC_OUT),
            fc_bias: vec![0.0; FC_OUT],
            bn_eps: 1e-5,
            arch: *arch,
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
        let ks = self.arch.kernel_size;
        let c1 = self.arch.conv1_out;
        let c2 = self.arch.conv2_out;
        anyhow::ensure!(self.conv1_weight.len() == c1 * EMBEDDING_DIM * ks);
        anyhow::ensure!(self.conv1_bias.len() == c1);
        anyhow::ensure!(self.bn1_gamma.len() == c1);
        anyhow::ensure!(self.bn1_beta.len() == c1);
        anyhow::ensure!(self.bn1_running_mean.len() == c1);
        anyhow::ensure!(self.bn1_running_var.len() == c1);
        anyhow::ensure!(self.conv2_weight.len() == c2 * c1 * ks);
        anyhow::ensure!(self.conv2_bias.len() == c2);
        anyhow::ensure!(self.bn2_gamma.len() == c2);
        anyhow::ensure!(self.bn2_beta.len() == c2);
        anyhow::ensure!(self.bn2_running_mean.len() == c2);
        anyhow::ensure!(self.bn2_running_var.len() == c2);
        anyhow::ensure!(self.fc_weight.len() == c2 * FC_OUT);
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
    /// entries and `forward()` averages their post-sigmoid scores uniformly.
    members: Vec<ClassifierWeights>,
}

// ── Unified forward pass ─────────────────────────────────────────────

/// Unified forward pass using frozen BN running statistics (mean=0, var=1).
///
/// This is the only forward path — there is no separate training pass
/// because batch-norm uses frozen identity statistics that act as a
/// learned per-channel affine transform, identical at train and inference
/// time.  No dropout is applied (the network has only 3 temporal positions,
/// making dropout destructively aggressive).
fn forward_pass(x: &[f32], w: &ClassifierWeights) -> f32 {
    let ks = w.arch.kernel_size;
    let c1 = w.arch.conv1_out;
    let c2 = w.arch.conv2_out;
    let mut h = conv1d(
        x,
        EMBEDDING_DIM,
        WINDOW_SIZE,
        c1,
        ks,
        &w.conv1_weight,
        &w.conv1_bias,
    );
    batch_norm(
        &mut h,
        c1,
        WINDOW_SIZE,
        &w.bn1_gamma,
        &w.bn1_beta,
        &w.bn1_running_mean,
        &w.bn1_running_var,
        w.bn_eps,
    );
    relu(&mut h);
    let mut h = conv1d(&h, c1, WINDOW_SIZE, c2, ks, &w.conv2_weight, &w.conv2_bias);
    batch_norm(
        &mut h,
        c2,
        WINDOW_SIZE,
        &w.bn2_gamma,
        &w.bn2_beta,
        &w.bn2_running_mean,
        &w.bn2_running_var,
        w.bn_eps,
    );
    relu(&mut h);
    let pooled = adaptive_avg_pool(&h, c2, WINDOW_SIZE);
    sigmoid(dot(&pooled, &w.fc_weight) + w.fc_bias[0])
}

impl WakeWordClassifier {
    /// Create a single-member classifier (legacy / backward compat path).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new(weights: ClassifierWeights) -> Self {
        Self {
            members: vec![weights],
        }
    }

    /// Return a reference to the ensemble member weights.
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub fn weights_ref(&self) -> &[ClassifierWeights] {
        &self.members
    }

    /// Create a multi-member ensemble classifier with uniform averaging.
    ///
    /// # Panics
    /// Panics if `members` is empty — an ensemble must have at least one model.
    pub fn new_ensemble(members: Vec<ClassifierWeights>) -> Self {
        assert!(
            !members.is_empty(),
            "Ensemble must have at least one member"
        );
        Self { members }
    }

    /// Run the forward pass through all ensemble members and return the
    /// uniformly-averaged post-sigmoid score (mahbot-904).
    ///
    /// For a single-model classifier this is equivalent to the original
    /// single-member forward pass.
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

        // Uniform average of post-sigmoid scores across all ensemble members
        // (mahbot-904).  All members contribute equally.
        let n = self.members.len() as f32;
        let mut total = 0.0;
        for w in &self.members {
            total += forward_pass(&cf, w) / n;
        }
        total
    }
}

// ── Forward primitives ──────────────────────────────────────────────────

fn conv1d(
    inp: &[f32],
    cin: usize,
    l: usize,
    cout: usize,
    kernel_size: usize,
    w: &[f32],
    b: &[f32],
) -> Vec<f32> {
    let padding = kernel_size / 2;
    let mut out = vec![0.0; cout * l];
    for co in 0..cout {
        for li in 0..l {
            let mut s = b[co];
            for ci in 0..cin {
                for k in 0..kernel_size {
                    let ii = li as isize + k as isize - padding as isize;
                    if ii >= 0 && ii < l as isize {
                        s += inp[ci * l + ii as usize] * w[(co * cin + ci) * kernel_size + k];
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
    /// Architecture configuration for this ensemble member (mahbot-848).
    /// Defines conv1_out, conv2_out, and kernel_size for the Conv1D layers.
    /// When `ArchConfig::default()` (baseline, 64/64/k3), behaviour matches
    /// the pre-848 single-architecture training.
    pub arch: ArchConfig,
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
            arch: ArchConfig::default(),
        }
    }
}

fn build_windows(sequences: &[EmbeddingSequence]) -> Vec<Vec<f32>> {
    let mut windows = Vec::new();
    for seq in sequences {
        if seq.embeddings.len() < WINDOW_SIZE {
            continue;
        }
        for i in 0..=(seq.embeddings.len() - WINDOW_SIZE) {
            let mut w = Vec::with_capacity(INPUT_DIM);
            for j in 0..WINDOW_SIZE {
                w.extend_from_slice(&seq.embeddings[i + j]);
            }
            windows.push(w);
        }
    }
    windows
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
    pos_sequences: &[EmbeddingSequence],
    neg_sequences: &[EmbeddingSequence],
    cfg: &TrainingConfig,
) -> Result<ClassifierTrainingResult> {
    let pos_w = build_windows(pos_sequences);
    let neg_w = build_windows(neg_sequences);
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

    // Train/val split — random 80/20 split (k-fold was removed in mahbot-851
    // because it reduced training data by 20% per ensemble member on a tiny
    // ~99-window dataset, preventing the model from learning).
    let n = all_x.len();
    let n_val_calc = ((n as f32) * cfg.validation_split).ceil() as usize;
    let n_val_calc = n_val_calc.max(1).min(n - 1);

    let mut rng: Box<dyn rand::Rng> = if let Some(seed) = cfg.rng_seed {
        Box::new(rand::rngs::StdRng::seed_from_u64(seed))
    } else {
        Box::new(rand::rng())
    };
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(&mut rng);

    let tr_x = gather(&all_x, &idx[n_val_calc..]);
    let tr_y = gather(&all_y, &idx[n_val_calc..]);
    let tr_w = gather(&all_w, &idx[n_val_calc..]);
    let va_x = gather(&all_x, &idx[..n_val_calc]);
    let va_y = gather(&all_y, &idx[..n_val_calc]);
    let va_w = gather(&all_w, &idx[..n_val_calc]);

    let n_tr = tr_x.len();
    let n_val = va_x.len();
    info!(
        "Training (random split): {n_tr} train + {n_val} val \
             ({np} pos + {nn} neg total)"
    );

    let cin = EMBEDDING_DIM;
    let lin = WINDOW_SIZE;
    let mut weights = ClassifierWeights::from_rng(&mut *rng, &cfg.arch);
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
            for &i in chunk {
                // Data augmentation removed in mahbot-851 — Gaussian noise
                // makes the learning problem harder on this tiny dataset
                // (~99 positive windows).  Train on raw windows.
                let x_cf = to_channels_first(&tr_x[i], cin, lin);
                let target = tr_y[i];
                let sw = tr_w[i];
                // Unified forward pass uses frozen BN running stats (mean=0, var=1)
                // acting as a learned affine transform — identical to inference.
                let pred = forward_pass(&x_cf, &weights);
                let eps = 1e-7;
                let loss =
                    -sw * (target * (pred + eps).ln() + (1.0 - target) * (1.0 - pred + eps).ln());
                batch_loss += loss;
                backward(&x_cf, target, sw, &weights, &mut g);
            }
            // Average gradients
            let nf = chunk.len() as f32;
            for gv in g.all_mut() {
                for v in gv {
                    *v /= nf;
                }
            }
            // Gradient clipping (mahbot-846): effectively disabled in mahbot-851
            // by setting GRADIENT_CLIP_NORM to 100.0 — on tiny datasets clipping
            // destroys the weak learning signal.
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
                let pred = forward_pass(&x_cf, &weights);
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
                forward_pass(&x_cf, &weights)
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

/// Manual backward pass using frozen BN running statistics (mean=0, var=1).
/// Accumulates gradients into `g`.
///
/// The gradient of binary cross-entropy w.r.t. the output logit is:
/// `d_logit = sw * (pred - target)`, where `sw` is the per-sample weight
/// (class-balanced by default).  Without the weight, negative samples in
/// an imbalanced dataset dominate the gradient — preventing the classifier
/// from learning meaningful class separation.
///
/// No per-sample batch statistics are used — BN running stats are always
/// the frozen identity (mean=0, var=1) that act as a learned per-channel
/// affine transform.  No dropout is applied (the network has only 3 temporal
/// positions, making dropout destructively aggressive).
#[allow(clippy::cast_precision_loss)]
fn backward(x: &[f32], target: f32, sw: f32, w: &ClassifierWeights, g: &mut GradientBuffer) {
    let cin = EMBEDDING_DIM;
    let lin = WINDOW_SIZE;
    let c1 = w.arch.conv1_out;
    let c2 = w.arch.conv2_out;
    let kernel_size = w.arch.kernel_size;
    let padding = kernel_size / 2;
    let eps = w.bn_eps;

    // Batch-norm always uses frozen running statistics (mean=0, var=1)
    // that act as a learned per-channel affine transform — identical at
    // train and inference time.
    let (bn1_mean, bn1_var, bn2_mean, bn2_var) = (
        &*w.bn1_running_mean,
        &*w.bn1_running_var,
        &*w.bn2_running_mean,
        &*w.bn2_running_var,
    );

    // Forward intermediates
    let mut conv1_pre = vec![0.0; c1 * lin];
    for co in 0..c1 {
        for li in 0..lin {
            let mut s = w.conv1_bias[co];
            for ci in 0..cin {
                for k in 0..kernel_size {
                    let ii = li as isize + k as isize - padding as isize;
                    if ii >= 0 && ii < lin as isize {
                        s += x[ci * lin + ii as usize]
                            * w.conv1_weight[(co * cin + ci) * kernel_size + k];
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

    let mut conv2_pre = vec![0.0; c2 * lin];
    for co in 0..c2 {
        for li in 0..lin {
            let mut s = w.conv2_bias[co];
            for ci in 0..c1 {
                for k in 0..kernel_size {
                    let ii = li as isize + k as isize - padding as isize;
                    if ii >= 0 && ii < lin as isize {
                        s += relu1[ci * lin + ii as usize]
                            * w.conv2_weight[(co * c1 + ci) * kernel_size + k];
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
    let d_logit = sw * (pred - target);

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
                for k in 0..kernel_size {
                    let ii = li as isize + k as isize - padding as isize;
                    if ii >= 0 && ii < lin as isize {
                        let widx = (co * c1 + ci) * kernel_size + k;
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
                for k in 0..kernel_size {
                    let ii = li as isize + k as isize - padding as isize;
                    if ii >= 0 && ii < lin as isize {
                        let widx = (co * cin + ci) * kernel_size + k;
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
        let arch = ArchConfig::default();
        let w = ClassifierWeights {
            conv1_weight: vec![0.0; arch.conv1_out * EMBEDDING_DIM * arch.kernel_size],
            conv1_bias: vec![0.0; arch.conv1_out],
            bn1_gamma: vec![1.0; arch.conv1_out],
            bn1_beta: vec![0.0; arch.conv1_out],
            bn1_running_mean: vec![0.0; arch.conv1_out],
            bn1_running_var: vec![1.0; arch.conv1_out],
            conv2_weight: vec![0.0; arch.conv2_out * arch.conv1_out * arch.kernel_size],
            conv2_bias: vec![0.0; arch.conv2_out],
            bn2_gamma: vec![1.0; arch.conv2_out],
            bn2_beta: vec![0.0; arch.conv2_out],
            bn2_running_mean: vec![0.0; arch.conv2_out],
            bn2_running_var: vec![1.0; arch.conv2_out],
            fc_weight: vec![0.0; arch.conv2_out * FC_OUT],
            fc_bias: vec![0.0; FC_OUT],
            bn_eps: 1e-5,
            arch,
        };
        let c = WakeWordClassifier::new(w);
        let score = c.forward(&embs);
        assert!((score - 0.5).abs() < 1e-4, "Expected 0.5, got {score}");
    }

    fn make_seq(embs: Vec<Vec<f32>>, label: LabelStratum) -> EmbeddingSequence {
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

    #[test]
    fn test_build_windows_basic() {
        let embs: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32; EMBEDDING_DIM]).collect();
        let seq = make_seq(embs, LabelStratum::Positive);
        assert_eq!(build_windows(&[seq]).len(), 3);
    }
    #[test]
    fn test_build_windows_empty() {
        assert!(build_windows(&[]).is_empty());
    }
    #[test]
    fn test_build_windows_short() {
        let embs: Vec<Vec<f32>> = (0..2).map(|i| vec![i as f32; EMBEDDING_DIM]).collect();
        let seq = make_seq(embs, LabelStratum::Positive);
        assert!(build_windows(&[seq]).is_empty());
    }

    #[test]
    fn test_build_windows_sequences_no_cross() {
        // Two sequences each shorter than WINDOW_SIZE → 0 windows
        // (no cross-sequence combination allowed).
        let embs1: Vec<Vec<f32>> = (0..2).map(|i| vec![i as f32; EMBEDDING_DIM]).collect();
        let embs2: Vec<Vec<f32>> = (0..2)
            .map(|i| vec![(i + 10) as f32; EMBEDDING_DIM])
            .collect();
        let seqs = [
            make_seq(embs1, LabelStratum::Positive),
            make_seq(embs2, LabelStratum::Positive),
        ];
        assert!(build_windows(&seqs).is_empty());
    }

    #[test]
    fn test_build_windows_two_sequences() {
        // Two sequences each exactly WINDOW_SIZE → 2 windows (1 per sequence).
        let embs1: Vec<Vec<f32>> = (0..WINDOW_SIZE)
            .map(|i| vec![i as f32; EMBEDDING_DIM])
            .collect();
        let embs2: Vec<Vec<f32>> = (0..WINDOW_SIZE)
            .map(|i| vec![(i + 100) as f32; EMBEDDING_DIM])
            .collect();
        let seqs = [
            make_seq(embs1, LabelStratum::Positive),
            make_seq(embs2, LabelStratum::Positive),
        ];
        assert_eq!(build_windows(&seqs).len(), 2);
    }

    #[test]
    fn test_embedding_sequence_metadata_preserved() {
        let embs: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32; EMBEDDING_DIM]).collect();
        let seq = EmbeddingSequence {
            id: crate::embedding_sequence::UtteranceId {
                sequence_index: 1,
                variant_index: 2,
            },
            source: crate::embedding_sequence::Source::Augmentation,
            augmentation_family: Some(crate::embedding_sequence::AugmentationFamily::Noise),
            label_stratum: LabelStratum::Positive,
            embeddings: embs,
        };
        assert_eq!(seq.id.sequence_index, 1);
        assert_eq!(seq.id.variant_index, 2);
        assert_eq!(seq.source, crate::embedding_sequence::Source::Augmentation);
        assert_eq!(
            seq.augmentation_family,
            Some(crate::embedding_sequence::AugmentationFamily::Noise)
        );
        assert_eq!(seq.label_stratum, LabelStratum::Positive);
    }

    #[test]
    fn test_weights_serde() {
        let w = ClassifierWeights::default();
        let json = serde_json::to_string(&w).unwrap();
        let _: ClassifierWeights = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_weights_serde_legacy() {
        // Legacy JSON without BN running stats fields and without the arch
        // field — these must be absent so that #[serde(default)] kicks in,
        // verifying backward compatibility with enrollment data serialized
        // before mahbot-846 added the four BN running-stat fields and
        // mahbot-848 added the per-member architecture configuration.
        let c1_def = DEFAULT_CONV1_OUT;
        let c2_def = DEFAULT_CONV2_OUT;
        let ks_def = DEFAULT_KERNEL_SIZE;
        let legacy = serde_json::json!({
            "conv1_weight": vec![0.0; c1_def * EMBEDDING_DIM * ks_def],
            "conv1_bias": vec![0.0; c1_def],
            "bn1_gamma": vec![1.0; c1_def],
            "bn1_beta": vec![0.0; c1_def],
            "conv2_weight": vec![0.0; c2_def * c1_def * ks_def],
            "conv2_bias": vec![0.0; c2_def],
            "bn2_gamma": vec![1.0; c2_def],
            "bn2_beta": vec![0.0; c2_def],
            "fc_weight": vec![0.0; c2_def * FC_OUT],
            "fc_bias": vec![0.0; FC_OUT],
            "bn_eps": 1e-5_f32,
        });
        let w: ClassifierWeights = serde_json::from_value(legacy).unwrap();
        // Verify BN running stats were defaulted to correct shapes and values.
        assert_eq!(w.bn1_running_mean.len(), c1_def);
        assert_eq!(w.bn1_running_var.len(), c1_def);
        assert_eq!(w.bn2_running_mean.len(), c2_def);
        assert_eq!(w.bn2_running_var.len(), c2_def);
        assert!(w.bn1_running_mean.iter().all(|&v| v == 0.0));
        assert!(w.bn1_running_var.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(w.bn2_running_mean.iter().all(|&v| v == 0.0));
        assert!(w.bn2_running_var.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        // Verify arch was defaulted to baseline (backward compat).
        assert_eq!(w.arch, ArchConfig::default());
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
        let pos_embs: Vec<Vec<f32>> = (0..n_embs).map(|_| make_emb(0.3, 0.4)).collect();
        let neg_embs: Vec<Vec<f32>> = (0..n_embs).map(|_| make_emb(-0.3, 0.4)).collect();
        // Each set is a single sequence to keep the window count the same.
        let pos_seqs = [make_seq(pos_embs, LabelStratum::Positive)];
        let neg_seqs = [make_seq(neg_embs, LabelStratum::Negative)];

        let cfg = TrainingConfig {
            max_epochs: 50,
            ..Default::default()
        };
        let result = train_classifier(&pos_seqs, &neg_seqs, &cfg).unwrap();
        let classifier = WakeWordClassifier::new_ensemble(result.weights);
        // Evaluate on the windows produced by build_windows — same path
        // that train_classifier uses internally.
        for win in build_windows(&pos_seqs) {
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
        for win in build_windows(&neg_seqs) {
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
    fn test_train_classifier_imbalanced() {
        // Imbalanced-data regression test (mahbot-925).
        //
        // With 1:30 class imbalance and WITHOUT the weighted gradient fix,
        // negative samples dominate the gradient ~30× more than intended,
        // causing the model to converge to near-zero (~0.15) predictions
        // with best validation loss ~0.93.
        //
        // With the fix, class-balanced weights restore gradient balance
        // (pw*pos_count == nw*neg_count), allowing meaningful class
        // separation with loss well below 0.93.
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut make_emb = |center: f32, noise: f32| -> Vec<f32> {
            (0..EMBEDDING_DIM)
                .map(|_| center + (rng.random::<f32>() - 0.5) * noise)
                .collect()
        };

        // build_windows with stride=1 produces N - WINDOW_SIZE + 1 windows
        // from a sequence of N embeddings.
        // 12 pos embeddings → 10 positive windows.
        // 302 neg embeddings → 300 negative windows (30× imbalance).
        let pos_embs: Vec<Vec<f32>> = (0..12).map(|_| make_emb(0.3, 0.4)).collect();
        let neg_embs: Vec<Vec<f32>> = (0..302).map(|_| make_emb(-0.3, 0.4)).collect();
        let pos_seqs = [make_seq(pos_embs, LabelStratum::Positive)];
        let neg_seqs = [make_seq(neg_embs, LabelStratum::Negative)];

        // Use a different seed for training split to avoid correlation with
        // the embedding generation seed.
        let cfg = TrainingConfig {
            max_epochs: 80,
            rng_seed: Some(99),
            ..Default::default()
        };
        let result = train_classifier(&pos_seqs, &neg_seqs, &cfg).unwrap();

        // Best validation loss must be significantly below the ~0.93 observed
        // without the weighted gradient (mahbot-925).  A model that fails to
        // learn (unweighted gradient) bottoms out around 0.93; a well-balanced
        // gradient should reach ≤0.69 (well below the chance-level ceiling).
        assert!(
            result.best_val_loss < 0.70,
            "Imbalanced training with weighted gradient should achieve \
             best val loss <0.70, got {}",
            result.best_val_loss,
        );

        // Evaluate on the windows produced by build_windows.
        let classifier = WakeWordClassifier::new_ensemble(result.weights);
        let mut pos_scores = Vec::new();
        for win in build_windows(&pos_seqs) {
            let embs: Vec<Vec<f32>> = (0..WINDOW_SIZE)
                .map(|t| {
                    let start = t * EMBEDDING_DIM;
                    win[start..start + EMBEDDING_DIM].to_vec()
                })
                .collect();
            pos_scores.push(classifier.forward(&embs));
        }
        let mut neg_scores = Vec::new();
        for win in build_windows(&neg_seqs) {
            let embs: Vec<Vec<f32>> = (0..WINDOW_SIZE)
                .map(|t| {
                    let start = t * EMBEDDING_DIM;
                    win[start..start + EMBEDDING_DIM].to_vec()
                })
                .collect();
            neg_scores.push(classifier.forward(&embs));
        }

        let pos_mean = pos_scores.iter().copied().sum::<f32>() / pos_scores.len() as f32;
        let neg_mean = neg_scores.iter().copied().sum::<f32>() / neg_scores.len() as f32;

        // With weighted gradients the positive mean must exceed the negative
        // mean by at least 0.3 — clear class separation despite 30:1 imbalance.
        assert!(
            pos_mean > neg_mean + 0.3,
            "Positive mean ({pos_mean:.4}) should exceed negative mean \
             ({neg_mean:.4}) by >0.3 with weighted gradient",
        );

        // Positive predictions should be above chance (0.5).
        assert!(
            pos_mean > 0.5,
            "Mean positive score should be >0.5, got {pos_mean:.4}",
        );

        // Negative predictions should be below chance (0.5).
        assert!(
            neg_mean < 0.5,
            "Mean negative score should be <0.5, got {neg_mean:.4}",
        );
    }

    #[test]
    fn test_classifier_deterministic_training() {
        // Two training runs with the same seed and identical training data
        // must produce identical weights (mahbot-904 AC #3).
        let mut rng = rand::rng();
        let mut make_emb = |center: f32, noise: f32| -> Vec<f32> {
            (0..EMBEDDING_DIM)
                .map(|_| center + (rng.random::<f32>() - 0.5) * noise)
                .collect()
        };

        // 30 windows each = 90 embeddings (WINDOW_SIZE per window).
        let n_wins = 30;
        let n_embs = n_wins * WINDOW_SIZE;
        let pos_embs: Vec<Vec<f32>> = (0..n_embs).map(|_| make_emb(0.3, 0.4)).collect();
        let neg_embs: Vec<Vec<f32>> = (0..n_embs).map(|_| make_emb(-0.3, 0.4)).collect();

        let pos_seq = make_seq(pos_embs, LabelStratum::Positive);
        let neg_seq = make_seq(neg_embs, LabelStratum::Negative);

        let cfg1 = TrainingConfig {
            rng_seed: Some(42),
            ..Default::default()
        };
        let cfg2 = TrainingConfig {
            rng_seed: Some(42),
            ..Default::default()
        };

        let result1 = train_classifier(&[pos_seq.clone()], &[neg_seq.clone()], &cfg1).unwrap();
        let result2 = train_classifier(&[pos_seq], &[neg_seq], &cfg2).unwrap();

        assert_eq!(
            result1.weights.len(),
            result2.weights.len(),
            "Ensemble member count must match"
        );
        for (i, (w1, w2)) in result1
            .weights
            .iter()
            .zip(result2.weights.iter())
            .enumerate()
        {
            // Compare all weight slices element-by-element.
            let all_slices_1 = w1.all_weight_slices();
            let all_slices_2 = w2.all_weight_slices();
            assert_eq!(
                all_slices_1.len(),
                all_slices_2.len(),
                "Weight slice count mismatch for member {i}",
            );
            for (j, (s1, s2)) in all_slices_1.iter().zip(all_slices_2.iter()).enumerate() {
                assert_eq!(
                    s1, s2,
                    "Weight slice {j} differs between deterministic training runs for member {i}",
                );
            }
        }
    }
}
