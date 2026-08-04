//! Conv1D wake word classifier — single member with reduced capacity.
//!
//! Architecture: Conv1D(96→4, k=3) + BN + ReLU → Conv1D(4→4, k=3) + BN + ReLU
//! → AdaptiveAvgPool1d → Linear(4→1) + Sigmoid.
//!
//! ~1.2K parameters (mahbot-931).  The 5-member ensemble with diverse
//! architectures was removed — a single small Conv1D trained on ~99 positive
//! windows captures temporal convolution patterns without ensemble overhead.
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

use crate::audio::embedding_sequence::EmbeddingSequence;

use crate::EMBEDDING_DIM;

// ── Constants ────────────────────────────────────────────────────────────

pub const WINDOW_SIZE: usize = 3;
pub const INPUT_DIM: usize = WINDOW_SIZE * EMBEDDING_DIM; // 288
/// Conv1D channel count after first convolution (96→4).
const CONV1_OUT: usize = 4;
/// Conv1D channel count after second convolution (4→4).
const CONV2_OUT: usize = 4;
const KERNEL_SIZE: usize = 3;
const FC_OUT: usize = 1;

/// Minimal architecture config — single member with ~1.2K params (mahbot-931).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ArchConfig {
    pub conv1_out: usize,
    pub conv2_out: usize,
    pub kernel_size: usize,
}

impl ArchConfig {
    #[must_use]
    pub const fn padding(&self) -> usize {
        self.kernel_size / 2
    }
}

impl Default for ArchConfig {
    fn default() -> Self {
        Self {
            conv1_out: CONV1_OUT,
            conv2_out: CONV2_OUT,
            kernel_size: KERNEL_SIZE,
        }
    }
}

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
const EARLY_STOP_PATIENCE: usize = 15;
const VALIDATION_SPLIT: f32 = 0.2;

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
    /// Architecture configuration for this classifier (mahbot-848).
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

    /// Mutable twin of [`all_trainable_slices`] — same canonical order.
    fn all_trainable_slices_mut(&mut self) -> [&mut [f32]; 10] {
        [
            &mut self.conv1_weight,
            &mut self.conv1_bias,
            &mut self.bn1_gamma,
            &mut self.bn1_beta,
            &mut self.conv2_weight,
            &mut self.conv2_bias,
            &mut self.bn2_gamma,
            &mut self.bn2_beta,
            &mut self.fc_weight,
            &mut self.fc_bias,
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
    /// Single Conv1D weight set (mahbot-931).  The 5-member ensemble was
    /// removed — a single small Conv1D (~1.2K params) captures temporal
    /// convolution patterns without ensemble overhead.
    weights: ClassifierWeights,
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
    /// Create a classifier from a single weight set.
    pub fn new(weights: ClassifierWeights) -> Self {
        Self { weights }
    }

    /// Return a reference to the classifier weights.
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub fn weights_ref(&self) -> &ClassifierWeights {
        &self.weights
    }

    /// Run the forward pass and return the post-sigmoid score.
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

        forward_pass(&cf, &self.weights)
    }
}

// ── Forward primitives ──────────────────────────────────────────────────

pub(crate) fn conv1d(
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

pub(crate) fn relu(x: &mut [f32]) {
    for v in x {
        *v = v.max(0.0);
    }
}

pub(crate) fn adaptive_avg_pool(x: &[f32], c: usize, l: usize) -> Vec<f32> {
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
    /// The trained Conv1D classifier weights.
    pub weights: ClassifierWeights,
    /// Actual number of epochs trained (may be less than
    /// `TrainingConfig::max_epochs` due to early stopping).
    pub epochs_trained: usize,
    /// Best validation loss achieved during training.
    pub best_val_loss: f32,
    /// Mean positive class score after training.
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub pos_scores_mean: f32,
    /// Minimum positive class score after training.
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub pos_scores_min: f32,
    /// Maximum positive class score after training.
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub pos_scores_max: f32,
    /// Mean negative class score after training.
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub neg_scores_mean: f32,
    /// Minimum negative class score after training.
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub neg_scores_min: f32,
    /// Maximum negative class score after training.
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub neg_scores_max: f32,
    /// Per-epoch training loss (cross-entropy over the training split),
    /// one entry per epoch actually trained (mahbot-1005 §5).
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub per_epoch_train_loss: Vec<f32>,
    /// Per-epoch validation loss, one entry per epoch actually trained.
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub per_epoch_val_loss: Vec<f32>,
    /// Per-epoch validation accuracy (pred > 0.5 matches the label),
    /// one entry per epoch actually trained.
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub per_epoch_val_accuracy: Vec<f32>,
    /// Why training stopped: `"max_epochs"` or `"early_stopping"`.
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub early_stop_reason: String,
    /// Number of training windows (after the random train/val split).
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub n_train_windows: usize,
    /// Number of validation windows (after the random train/val split).
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub n_val_windows: usize,
    /// Decile boundaries of the training-set positive-window scores
    /// (mahbot-1005 §3).  `None` when no positive windows existed.
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub pos_scores_deciles: Option<[f32; 10]>,
    /// Decile boundaries of the training-set negative-window scores
    /// (mahbot-1005 §3).
    #[cfg_attr(not(feature = "voice-tests"), allow(dead_code))]
    pub neg_scores_deciles: Option<[f32; 10]>,
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
    /// Architecture configuration for this classifier (mahbot-848).
    /// Defines conv1_out, conv2_out, and kernel_size for the Conv1D layers.
    /// When `ArchConfig::default()` (baseline, 4/4/k3), behaviour matches
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
        if seq.embeddings.is_empty() {
            continue;
        }
        if seq.embeddings.len() >= WINDOW_SIZE {
            // Normal sliding windows over consecutive embeddings.
            for i in 0..=(seq.embeddings.len() - WINDOW_SIZE) {
                let mut w = Vec::with_capacity(INPUT_DIM);
                for j in 0..WINDOW_SIZE {
                    w.extend_from_slice(&seq.embeddings[i + j]);
                }
                windows.push(w);
            }
        } else {
            // Short sequence (< WINDOW_SIZE embeddings): simulate the
            // streaming cold-start ring-buffer accumulation (mahbot-1001 Fix 4).
            //
            // Streaming inference in score_single_embedding builds the classifier
            // window by tiling the current ring buffer with repeat-last:
            //   window[j] = ring[min(j, ring.len() - 1)]
            //
            // As each new embedding arrives, the ring grows from [e0] to [e0, e1]
            // and produces windows that are never seen during normal sliding-window
            // training.  Without matching these in training, the Conv1D classifier
            // receives out-of-distribution windows at cold-start and correctly
            // rejects them with near-zero sigmoid scores.
            //
            // For a sequence of N embeddings (N < WINDOW_SIZE), simulate the
            // ring state after each step k (0 ≤ k < N):
            //   ring = embeddings[0..=k]  (k+1 entries piled up)
            //   last = k
            //   window[j] = embeddings[min(j, k)]
            for k in 0..seq.embeddings.len() {
                let mut w = Vec::with_capacity(INPUT_DIM);
                for j in 0..WINDOW_SIZE {
                    let src_idx = j.min(k);
                    w.extend_from_slice(&seq.embeddings[src_idx]);
                }
                windows.push(w);
            }
        }
    }
    windows
}

pub(crate) struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
}
impl AdamState {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        }
    }
    pub(crate) fn update(&mut self, p: &mut [f32], g: &[f32], lr: f32) {
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
    // because it reduced training data by 20% on a tiny ~99-window dataset,
    // preventing the model from learning).
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
    // Per-epoch diagnostics (mahbot-1005 §5) — previously computed and logged
    // but discarded.  The vectors cap at `cfg.max_epochs` (100) entries.
    let mut per_epoch_train_loss: Vec<f32> = Vec::new();
    let mut per_epoch_val_loss: Vec<f32> = Vec::new();
    let mut per_epoch_val_accuracy: Vec<f32> = Vec::new();
    let mut early_stop_reason = "max_epochs".to_string();

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
            opt.step(&mut weights, &mut g, lr);
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

        // Validation accuracy: fraction of val windows where pred > 0.5
        // matches the label (mahbot-1005 §5).
        let val_accuracy = if va_x.is_empty() {
            0.0
        } else {
            let mut correct = 0usize;
            for i in 0..va_x.len() {
                let x_cf = to_channels_first(&va_x[i], cin, lin);
                let pred = forward_pass(&x_cf, &weights);
                if (pred > 0.5) == (va_y[i] > 0.5) {
                    correct += 1;
                }
            }
            correct as f32 / va_x.len() as f32
        };
        per_epoch_train_loss.push(epoch_loss / n_batches as f32);
        per_epoch_val_loss.push(val_loss);
        per_epoch_val_accuracy.push(val_accuracy);

        info!(
            "Epoch {}/{}: loss={:.6} val={:.6} val_acc={:.3}",
            epoch + 1,
            cfg.max_epochs,
            epoch_loss / n_batches as f32,
            val_loss,
            val_accuracy,
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
                early_stop_reason = "early_stopping".to_string();
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
        weights,
        epochs_trained,
        best_val_loss: best_loss,
        pos_scores_mean,
        pos_scores_min,
        pos_scores_max,
        neg_scores_mean,
        neg_scores_min,
        neg_scores_max,
        per_epoch_train_loss,
        per_epoch_val_loss,
        per_epoch_val_accuracy,
        early_stop_reason,
        n_train_windows: n_tr,
        n_val_windows: n_val,
        pos_scores_deciles: score_deciles(&pos_scores),
        neg_scores_deciles: score_deciles(&neg_scores),
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

/// Decile boundaries of a score distribution (`None` for empty input).
/// Added for benchmark training-score diagnostics (mahbot-1005 §3).
fn score_deciles(scores: &[f32]) -> Option<[f32; 10]> {
    if scores.is_empty() {
        return None;
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = [0.0_f32; 10];
    for (i, slot) in out.iter_mut().enumerate() {
        let idx = ((sorted.len() - 1) as f32 * (i as f32 + 0.5) / 10.0).round() as usize;
        *slot = sorted[idx.min(sorted.len() - 1)];
    }
    Some(out)
}

fn gather<T: Clone>(data: &[T], idx: &[usize]) -> Vec<T> {
    idx.iter().map(|&i| data[i].clone()).collect()
}

pub(crate) fn to_channels_first(x: &[f32], cin: usize, lin: usize) -> Vec<f32> {
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
// NOTE: Adding a new weight tensor requires updating the canonical
// enumerations (ClassifierWeights `all_trainable_slices`/_mut, GradientBuffer
// and AdamStateGroup `all_mut`) plus named-field consumers (`validate()`,
// `Default for ClassifierWeights`, `param_count`, L2 loops).  `new`/`step`
// consume the accessors positionally — arity-checked destructuring turns a
// field-count drift into a compile error instead of silent gradient omission,
// at the cost of one 10-field enumeration per accessor (net +37 lines vs the
// old duplicated sites).  `step` borrows `g` mutably only to reuse the single
// GradientBuffer accessor; the gradients themselves are read-only there.

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
        let [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10] = w.all_trainable_slices();
        Self {
            conv1_w: vec![0.0; c1.len()],
            conv1_b: vec![0.0; c2.len()],
            bn1_gamma: vec![0.0; c3.len()],
            bn1_beta: vec![0.0; c4.len()],
            conv2_w: vec![0.0; c5.len()],
            conv2_b: vec![0.0; c6.len()],
            bn2_gamma: vec![0.0; c7.len()],
            bn2_beta: vec![0.0; c8.len()],
            fc_w: vec![0.0; c9.len()],
            fc_b: vec![0.0; c10.len()],
        }
    }
    /// Single enumeration of the 10 trainable gradient buffers (canonical order).
    fn all_mut(&mut self) -> [&mut [f32]; 10] {
        [
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
        let [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10] = w.all_trainable_slices();
        Self {
            conv1_w: AdamState::new(c1.len()),
            conv1_b: AdamState::new(c2.len()),
            bn1_gamma: AdamState::new(c3.len()),
            bn1_beta: AdamState::new(c4.len()),
            conv2_w: AdamState::new(c5.len()),
            conv2_b: AdamState::new(c6.len()),
            bn2_gamma: AdamState::new(c7.len()),
            bn2_beta: AdamState::new(c8.len()),
            fc_w: AdamState::new(c9.len()),
            fc_b: AdamState::new(c10.len()),
        }
    }
    /// Single enumeration of the 10 Adam states (canonical order).
    fn all_mut(&mut self) -> [&mut AdamState; 10] {
        [
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
    fn step(&mut self, w: &mut ClassifierWeights, g: &mut GradientBuffer, lr: f32) {
        let [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] = self.all_mut();
        let [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10] = w.all_trainable_slices_mut();
        let [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10] = g.all_mut();
        s1.update(w1, g1, lr);
        s2.update(w2, g2, lr);
        s3.update(w3, g3, lr);
        s4.update(w4, g4, lr);
        s5.update(w5, g5, lr);
        s6.update(w6, g6, lr);
        s7.update(w7, g7, lr);
        s8.update(w8, g8, lr);
        s9.update(w9, g9, lr);
        s10.update(w10, g10, lr);
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
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    // Shared test fixture (mahbot-1043): single authoritative make_seq body
    // lives next to the EmbeddingSequence type; local builders drifted once.
    use crate::audio::embedding_sequence::make_test_sequence as make_seq;
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

    #[test]
    fn test_build_windows_basic() {
        let embs: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32; EMBEDDING_DIM]).collect();
        let seq = make_seq(embs);
        assert_eq!(build_windows(&[seq]).len(), 3);

        // Empty array produces no windows.
        assert!(build_windows(&[]).is_empty());

        // Short array (< WINDOW_SIZE embeddings) produces tiled windows
        // matching streaming inference behavior (mahbot-1001 Fix 4).
        // 2 embeddings → 2 tiled windows: [e0, e0, e0] and [e0, e1, e1]
        let short_embs: Vec<Vec<f32>> = (0..2).map(|i| vec![i as f32; EMBEDDING_DIM]).collect();
        let short_seq = make_seq(short_embs);
        assert_eq!(build_windows(&[short_seq]).len(), 2);
    }

    #[test]
    fn test_build_windows_sequences_no_cross() {
        // Two sequences each shorter than WINDOW_SIZE → each produces
        // tiled windows (no cross-sequence combination).  2 embeddings
        // each → 2 + 2 = 4 tiled windows.
        let embs1: Vec<Vec<f32>> = (0..2).map(|i| vec![i as f32; EMBEDDING_DIM]).collect();
        let embs2: Vec<Vec<f32>> = (0..2)
            .map(|i| vec![(i + 10) as f32; EMBEDDING_DIM])
            .collect();
        let seqs = [make_seq(embs1), make_seq(embs2)];
        assert!(
            !build_windows(&seqs).is_empty(),
            "short sequences now produce tiled windows (mahbot-1001)"
        );

        // Two sequences each exactly WINDOW_SIZE → 2 windows (1 per sequence).
        let embs3: Vec<Vec<f32>> = (0..WINDOW_SIZE)
            .map(|i| vec![i as f32; EMBEDDING_DIM])
            .collect();
        let embs4: Vec<Vec<f32>> = (0..WINDOW_SIZE)
            .map(|i| vec![(i + 100) as f32; EMBEDDING_DIM])
            .collect();
        let seqs2 = [make_seq(embs3), make_seq(embs4)];
        assert_eq!(build_windows(&seqs2).len(), 2);
    }

    #[test]
    fn test_weights_serde() {
        let w = ClassifierWeights::default();
        // Default weights should pass validation.
        assert!(w.validate().is_ok());
        let json = serde_json::to_string(&w).unwrap();
        let deserialized: ClassifierWeights = serde_json::from_str(&json).unwrap();
        // Round-tripped weights should also pass validation.
        assert!(deserialized.validate().is_ok());
    }

    #[test]
    fn test_weights_serde_legacy() {
        // Legacy JSON without BN running stats fields and without the arch
        // field — these must be absent so that #[serde(default)] kicks in,
        // verifying backward compatibility with enrollment data serialized
        // before mahbot-846 added the four BN running-stat fields and
        // mahbot-848 added the per-member architecture configuration.
        let c1_def = CONV1_OUT;
        let c2_def = CONV2_OUT;
        let ks_def = KERNEL_SIZE;
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
        // Use seeded RNG for deterministic data generation.
        let mut rng = StdRng::seed_from_u64(42);
        let mut make_emb = |center: f32, noise: f32| -> Vec<f32> {
            (0..EMBEDDING_DIM)
                .map(|_| center + (rng.random::<f32>() - 0.5) * noise)
                .collect()
        };

        // 60 windows each = 180 embeddings (WINDOW_SIZE per window).
        // (Reduced from 100 windows / 300 embeddings in mahbot-1029 — the
        // separable clusters converge with a wide margin at these sizes.)
        let n_wins = 60;
        let n_embs = n_wins * WINDOW_SIZE;
        let pos_embs: Vec<Vec<f32>> = (0..n_embs).map(|_| make_emb(0.3, 0.4)).collect();
        let neg_embs: Vec<Vec<f32>> = (0..n_embs).map(|_| make_emb(-0.3, 0.4)).collect();
        // Each set is a single sequence to keep the window count the same.
        let pos_seqs = [make_seq(pos_embs)];
        let neg_seqs = [make_seq(neg_embs)];

        let cfg = TrainingConfig {
            // Reduced from 80 in mahbot-1029 — convergence margin validated
            // at this size (well-separated ±0.3 clusters).
            max_epochs: 50,
            rng_seed: Some(42),
            ..Default::default()
        };
        let result = train_classifier(&pos_seqs, &neg_seqs, &cfg).unwrap();
        let weights_clone = result.weights.clone();
        let classifier = WakeWordClassifier::new(result.weights);

        // ── Convergence assertions ──────────────────────────────────
        // Evaluate on the windows produced by build_windows — same path
        // that train_classifier uses internally.
        let mut pos_pass = 0;
        let mut pos_total = 0;
        for win in build_windows(&pos_seqs) {
            let embs: Vec<Vec<f32>> = (0..WINDOW_SIZE)
                .map(|t| {
                    let start = t * EMBEDDING_DIM;
                    win[start..start + EMBEDDING_DIM].to_vec()
                })
                .collect();
            let score = classifier.forward(&embs);
            pos_total += 1;
            if score > 0.8 {
                pos_pass += 1;
            }
            assert!(
                score > 0.6,
                "Positive window should score >0.6, got {score}"
            );
        }
        assert!(
            pos_pass as f64 / pos_total as f64 > 0.8,
            "At least 80% of positive windows should score >0.8, got {pos_pass}/{pos_total}"
        );
        for win in build_windows(&neg_seqs) {
            let embs: Vec<Vec<f32>> = (0..WINDOW_SIZE)
                .map(|t| {
                    let start = t * EMBEDDING_DIM;
                    win[start..start + EMBEDDING_DIM].to_vec()
                })
                .collect();
            let score = classifier.forward(&embs);
            assert!(
                score < 0.4,
                "Negative window should score <0.4, got {score}"
            );
        }

        // ── Imbalanced weighted-gradient assertions ─────────────────
        // With weighted gradients the positive mean must exceed the negative
        // mean by at least 0.3 — clear class separation.
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

        assert!(
            pos_mean > neg_mean + 0.3,
            "Positive mean ({pos_mean:.4}) should exceed negative mean \
             ({neg_mean:.4}) by >0.3",
        );

        assert!(
            pos_mean > 0.5,
            "Mean positive score should be >0.5, got {pos_mean:.4}",
        );

        assert!(
            neg_mean < 0.5,
            "Mean negative score should be <0.5, got {neg_mean:.4}",
        );

        // Best validation loss must be below ~0.93 (the ceiling without
        // weighted gradients). With balanced data this is even lower.
        assert!(
            result.best_val_loss < 0.70,
            "Training should achieve best val loss <0.70, got {}",
            result.best_val_loss,
        );

        // ── Deterministic reproducibility assertions ────────────────
        // A second training run with the same seed must produce identical weights.
        let result2 = train_classifier(&pos_seqs, &neg_seqs, &cfg).unwrap();

        assert_eq!(
            weights_clone.all_weight_slices().len(),
            result2.weights.all_weight_slices().len(),
            "Weight slice count must match between deterministic runs"
        );
        for (s1, s2) in weights_clone
            .all_weight_slices()
            .iter()
            .zip(result2.weights.all_weight_slices().iter())
        {
            assert_eq!(s1, s2, "Weights differ between deterministic runs");
        }
    }
}
