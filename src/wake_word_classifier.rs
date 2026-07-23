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
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::voice_verifier::EMBEDDING_DIM;

// ── Constants ────────────────────────────────────────────────────────────

pub const WINDOW_SIZE: usize = 3;
pub const INPUT_DIM: usize = WINDOW_SIZE * EMBEDDING_DIM; // 288
const CONV1_OUT: usize = 64;
const CONV2_OUT: usize = 64;
const KERNEL_SIZE: usize = 3;
const PADDING: usize = 1;
const FC_OUT: usize = 1;
const L2_LAMBDA: f32 = 0.001;
const LEARNING_RATE: f32 = 0.001;
const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;
const BATCH_SZ: usize = 32;
const MAX_EPOCHS: usize = 100;
const EARLY_STOP_PATIENCE: usize = 5;
const VALIDATION_SPLIT: f32 = 0.2;

// ── Weights ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierWeights {
    pub conv1_weight: Vec<f32>, // [64, 96, 3]
    pub conv1_bias: Vec<f32>,   // [64]
    pub bn1_gamma: Vec<f32>,
    pub bn1_beta: Vec<f32>,
    pub bn1_running_mean: Vec<f32>,
    pub bn1_running_var: Vec<f32>,
    pub conv2_weight: Vec<f32>, // [64, 64, 3]
    pub conv2_bias: Vec<f32>,   // [64]
    pub bn2_gamma: Vec<f32>,
    pub bn2_beta: Vec<f32>,
    pub bn2_running_mean: Vec<f32>,
    pub bn2_running_var: Vec<f32>,
    pub fc_weight: Vec<f32>, // [1, 64]
    pub fc_bias: Vec<f32>,   // [1]
    pub bn_eps: f32,
}

impl Default for ClassifierWeights {
    fn default() -> Self {
        let scale_c1 = (6.0 / (EMBEDDING_DIM + CONV1_OUT) as f32).sqrt();
        let scale_c2 = (6.0 / (CONV1_OUT + CONV2_OUT) as f32).sqrt();
        let scale_fc = (6.0 / (CONV2_OUT + FC_OUT) as f32).sqrt();
        let mut rng = rand::rng();
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
}

impl ClassifierWeights {
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
            self.conv1_weight
                .iter()
                .chain(self.conv1_bias.iter())
                .chain(self.bn1_gamma.iter())
                .chain(self.bn1_beta.iter())
                .chain(self.bn1_running_mean.iter())
                .chain(self.bn1_running_var.iter())
                .chain(self.conv2_weight.iter())
                .chain(self.conv2_bias.iter())
                .chain(self.bn2_gamma.iter())
                .chain(self.bn2_beta.iter())
                .chain(self.bn2_running_mean.iter())
                .chain(self.bn2_running_var.iter())
                .chain(self.fc_weight.iter())
                .chain(self.fc_bias.iter())
                .all(|v| v.is_finite()),
            "Classifier weights contain NaN or Infinity"
        );
        Ok(())
    }
    pub fn param_count(&self) -> usize {
        self.conv1_weight.len()
            + self.conv1_bias.len()
            + self.bn1_gamma.len()
            + self.bn1_beta.len()
            + self.bn1_running_mean.len()
            + self.bn1_running_var.len()
            + self.conv2_weight.len()
            + self.conv2_bias.len()
            + self.bn2_gamma.len()
            + self.bn2_beta.len()
            + self.bn2_running_mean.len()
            + self.bn2_running_var.len()
            + self.fc_weight.len()
            + self.fc_bias.len()
    }
}

// ── Classifier ──────────────────────────────────────────────────────────

pub struct WakeWordClassifier {
    weights: ClassifierWeights,
}

/// Run the full Conv1D→BN→ReLU→Conv1D→BN→ReLU→AvgPool→FC→Sigmoid forward pass.
///
/// Input `x` must be in channels-first layout (shape `[EMBEDDING_DIM, WINDOW_SIZE]`).
///
/// NOTE: During training, batch norm running mean/var are NOT updated — the BN
/// layers act as learned affine transforms with fixed statistics. This
/// simplifies training without materially affecting accuracy for a binary
/// wake-word classifier with stationary input distribution.
fn forward_pass(x: &[f32], w: &ClassifierWeights) -> f32 {
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

impl WakeWordClassifier {
    /// Get a reference to the underlying classifier weights.
    #[expect(dead_code)]
    pub fn weights_ref(&self) -> &ClassifierWeights {
        &self.weights
    }

    pub fn new(weights: ClassifierWeights) -> Self {
        Self { weights }
    }

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

// ── Training ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub l2_lambda: f32,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub early_stop_patience: usize,
    pub validation_split: f32,
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
) -> Result<ClassifierWeights> {
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

    // Train/val split
    let n = all_x.len();
    let n_val = ((n as f32) * cfg.validation_split).ceil() as usize;
    let n_val = n_val.max(1).min(n - 1);
    let n_tr = n - n_val;

    let mut rng = rand::rng();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(&mut rng);

    let tr_x = gather(&all_x, &idx[n_val..]);
    let tr_y = gather(&all_y, &idx[n_val..]);
    let tr_w = gather(&all_w, &idx[n_val..]);
    let va_x = gather(&all_x, &idx[..n_val]);
    let va_y = gather(&all_y, &idx[..n_val]);
    let va_w = gather(&all_w, &idx[..n_val]);

    info!("Training: {n_tr} train + {n_val} val ({np} pos + {nn} neg total)");

    let cin = EMBEDDING_DIM;
    let lin = WINDOW_SIZE;
    let mut weights = ClassifierWeights::default();
    let bs = cfg.batch_size.min(n_tr).max(1);
    let mut best_loss = f32::INFINITY;
    let mut patience = 0;
    let mut best = weights.clone();

    let mut opt = AdamStateGroup::new(&weights);

    for epoch in 0..cfg.max_epochs {
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
                let x_cf = to_channels_first(&tr_x[i], cin, lin);
                let target = tr_y[i];
                let sw = tr_w[i];
                let pred = forward_pass(&x_cf, &weights);
                let eps = 1e-7;
                let loss =
                    -sw * (target * (pred + eps).ln() + (1.0 - target) * (1.0 - pred + eps).ln());
                batch_loss += loss;
                backward(&x_cf, target, &weights, &mut g);
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

        debug!(
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
    weights.validate()?;
    Ok(weights)
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
#[allow(clippy::cast_precision_loss)]
fn backward(x: &[f32], target: f32, w: &ClassifierWeights, g: &mut GradientBuffer) {
    let cin = EMBEDDING_DIM;
    let lin = WINDOW_SIZE;
    let c1 = CONV1_OUT;
    let c2 = CONV2_OUT;
    let eps = w.bn_eps;

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
        let std = (w.bn1_running_var[ci] + eps).sqrt();
        bn1_std[ci] = std;
        for li in 0..lin {
            let idx = ci * lin + li;
            bn1_xhat[idx] = (conv1_pre[idx] - w.bn1_running_mean[ci]) / std;
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
        let std = (w.bn2_running_var[ci] + eps).sqrt();
        bn2_std[ci] = std;
        for li in 0..lin {
            let idx = ci * lin + li;
            bn2_xhat[idx] = (conv2_pre[idx] - w.bn2_running_mean[ci]) / std;
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
        let w = train_classifier(&pos, &neg, &cfg).unwrap();

        let classifier = WakeWordClassifier::new(w);
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
}
