//! Wake word detection on the shared Qwen3-ASR encoder.
//!
//! # Architecture
//!
//! The wake-word pipeline is built entirely on the Qwen3-ASR encoder features:
//! no separate embedding model, no trainable head, no AGC/NS preprocessing.
//! It reuses the exact same loaded ASR model instance as speech-to-text
//! transcription ([`crate::audio::local_transcriber::shared_model_arc`]).
//!
//! Feature extraction (per bounded ≤1 s window):
//! 1. Take the trailing [`WINDOW_SAMPLES`] (12 160 samples ≈ 0.76 s) of mono
//!    16 kHz audio.
//! 2. Run the encoder's mel frontend (`qwen_asr::audio::mel_spectrogram`,
//!    128 bands, 10 ms stride → [`WINDOW_MEL_FRAMES`] mel frames).
//! 3. Run the encoder transformer (`Encoder::forward`) → 10 tokens × 1024 dim.
//! 4. Mean-pool the 10 tokens → one 1024-dim window embedding, L2-normalized.
//!
//! The encoder has no streaming state: every scoring step is an independent
//! bounded-window forward.  No incremental/streaming encoder machinery exists.
//!
//! # Enrollment (prototype/centroid, no trainable head)
//!
//! ~10 utterances are each embedded via the same window pipeline.  The
//! prototype is the L2-normalized mean of the accepted utterance embeddings.
//! Negative-sample calibration (owner negatives + ambient) produces a cosine
//! floor used to map raw cosine similarity into the soft-score space consumed
//! by the rolling/adaptive machinery in `voice.rs`, plus anti-prototypes that
//! gate out windows closer to a negative than to the wake word.
//!
//! # Persistence
//!
//! Schema version 2 (`wake_word_templates` config key): phrase, 1024-dim
//! prototype, calibration stats, anti-prototypes, timestamps.  Old v1
//! enrollment records are rejected outright — no migration, no compat.
//!
//! # Mel-normalization consistency
//!
//! `qwen_asr::audio::mel_spectrogram` normalizes each call by its global max.
//! Both enrollment and streaming therefore always feed exactly the same
//! window shape — the trailing [`WINDOW_SAMPLES`] samples of VAD-gated speech
//! — so the per-call normalization is identical by construction.  The
//! voice-pipeline benchmark validates this explicitly (mel-normalization
//! consistency probe: the same clip encoded whole-utterance vs
//! streaming-accumulated must yield near-identical embeddings).

use anyhow::{Result, anyhow};
use qwen_asr::context::QwenModel;
use serde::{Deserialize, Serialize};

use crate::vector::cosine_similarity;

// ── Constants ────────────────────────────────────────────────────────────

/// Dimensionality of the projected encoder features (Qwen3-ASR 0.6B:
/// `enc_output_dim` = decoder hidden = 1024).
pub(crate) const WAKE_WORD_EMBEDDING_DIM: usize = 1024;

/// Bounded detection window: 76 mel frames ≈ 0.76 s at 16 kHz (12 160 samples).
///
/// Chosen over a full 1 s window: the encoder's conv stem downsamples 100
/// frames to 13 tokens, but a 1 s window includes ~0.3-0.5 s of trailing
/// silence/context for a typical ~0.5 s phrase, diluting the mean-pooled
/// embedding and collapsing confusable discrimination (measured: 40-clip
/// pair probe finds 25% of pairs at cosine >0.85 with a 1 s window vs 12%
/// with 76 frames).  76 frames still yields ~10 tokens per window.
pub(crate) const WINDOW_MEL_FRAMES: usize = 76;

/// Detection window in samples: [`WINDOW_MEL_FRAMES`] × 160-sample hop.
pub(crate) const WINDOW_SAMPLES: usize = WINDOW_MEL_FRAMES * 160;

/// Maximum encoder tokens any window can produce: a full 100-frame chunk
/// yields 13 tokens (conv-stem output sizes w1=50, w2=25, w3=13).  A
/// [`WINDOW_MEL_FRAMES`]-frame (76) window yields only 10
/// (w1=38, w2=19, w3=10) — this constant is the upper bound used by the
/// `debug_assert` in [`encode_window`], not the per-window token count.
const MAX_TOKENS_PER_CHUNK: usize = 13;

/// Scoring stride in mel frames (stride 16 ≈ 160 ms).  The old pipeline used
/// stride 8 over 76-frame windows; the encoder forward is far heavier, so the
/// stride is widened to keep the real-time budget sane while the rolling
/// window (N=3) still spans ~0.5 s of speech.
pub(crate) const SCORE_STRIDE_MEL_FRAMES: usize = 16;

/// Maximum cosine floor — the upper clamp bound for the calibrated floor.
/// The soft-score mapping anchors the floor at score 0, so a floor set
/// at/above a match's cosine would squash that match to zero; this constant
/// keeps the calibrated floor below the match range.
const MAX_SOFT_FLOOR: f32 = 0.75;

/// Fallback cosine floor when no negative calibration exists (e.g. the
/// enrollment skipped negatives).
const DEFAULT_SOFT_FLOOR: f32 = 0.55;

/// Enrollment consistency: minimum cosine between an utterance embedding and
/// the centroid, and the fraction of utterances that must pass.
pub(crate) const ENROLLMENT_CONSISTENCY_MIN_SIMILARITY: f32 = 0.70;
pub(crate) const ENROLLMENT_CONSISTENCY_MIN_FRACTION: f32 = 0.7;

/// Minimum number of qualified utterances for a valid enrollment.
pub(crate) const MIN_ENROLLMENT_UTTERANCES: usize = 5;

// ── Feature extraction ───────────────────────────────────────────────────

/// Embed a bounded ≤1 s window of mono 16 kHz audio through the shared
/// Qwen3-ASR encoder.
///
/// Takes the **trailing** [`WINDOW_SAMPLES`] samples (zero-padded at the
/// front when the input is shorter) so enrollment and streaming always feed
/// the encoder the same window shape — the mel frontend's per-call global-max
/// normalization is then identical by construction.
///
/// Returns an L2-normalized [`WAKE_WORD_EMBEDDING_DIM`]-dim embedding.
///
/// # Thread safety
///
/// The encoder weights are read-only; `enc_bufs = None` allocates fresh
/// per-call scratch, so this can run concurrently with transcription on the
/// same model Arc.
pub(crate) fn encode_window(model: &QwenModel, samples: &[f32]) -> Result<Vec<f32>> {
    // ── Trailing window, zero-padded at the front ──
    let take = samples.len().min(WINDOW_SAMPLES);
    let mut window = vec![0.0; WINDOW_SAMPLES];
    window[WINDOW_SAMPLES - take..].copy_from_slice(&samples[samples.len() - take..]);

    // ── Mel frontend (128 bands, 10 ms stride) ──
    let (mel, mel_frames) = qwen_asr::audio::mel_spectrogram(&window).ok_or_else(|| {
        anyhow!("mel_spectrogram returned None for {WINDOW_SAMPLES}-sample window")
    })?;
    debug_assert_eq!(mel_frames, WINDOW_MEL_FRAMES);

    // ── Encoder forward (fresh buffers — safe concurrent with transcription) ──
    let (features, total_tokens) = model
        .encoder
        .forward(&model.config, &mel, mel_frames, None)
        .ok_or_else(|| anyhow!("encoder forward failed for {mel_frames}-frame window"))?;
    debug_assert!(total_tokens <= MAX_TOKENS_PER_CHUNK);

    // ── Mean-pool tokens → window embedding ──
    let output_dim = model.config.enc_output_dim;
    let mut pooled = vec![0.0f32; output_dim];
    for t in 0..total_tokens {
        let base = t * output_dim;
        for (d, v) in pooled.iter_mut().enumerate() {
            *v += features[base + d];
        }
    }
    #[expect(clippy::cast_precision_loss)]
    let inv = 1.0 / total_tokens as f32;
    for v in &mut pooled {
        *v *= inv;
    }

    l2_normalize_in_place(&mut pooled);
    Ok(pooled)
}

/// L2-normalize a vector (returns a new vector; no-op for zero/empty input).
/// Test-only helper — production paths use [`l2_normalize_in_place`].
#[cfg(test)]
#[must_use]
pub(crate) fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let mut out = v.to_vec();
    l2_normalize_in_place(&mut out);
    out
}

/// L2-normalize a vector in place (no-op for zero/empty vectors).
fn l2_normalize_in_place(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

// ── Persistence schema (v2) ─────────────────────────────────────────────

/// Serialization schema version for [`WakeWordEnrollment`].  v1 (classifier)
/// records are rejected at load — there is no migration path.
pub(crate) const ENROLLMENT_SCHEMA_VERSION: u32 = 2;

/// Negative-sample calibration stats: cosine distribution of negative
/// material against the prototype, measured at enrollment time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Calibration {
    /// Mean cosine of negative samples vs the prototype.
    pub(crate) neg_mean: f32,
    /// Standard deviation of the negative cosine distribution.
    pub(crate) neg_std: f32,
    /// 99th percentile of the negative cosine distribution — the calibration
    /// floor source ([`Calibration::soft_floor`]).
    pub(crate) neg_p99: f32,
    /// Number of negative windows scored.
    pub(crate) n_negatives: usize,
}

impl Calibration {
    /// Cosine floor for the soft-score mapping: `neg_p99` clamped to
    /// [`DEFAULT_SOFT_FLOOR`]..=[`MAX_SOFT_FLOOR`].  Scores below the floor map
    /// to ~0, so the negative material sits at the bottom of the soft-score
    /// range.
    #[must_use]
    pub(crate) fn soft_floor(&self) -> f32 {
        self.neg_p99.clamp(DEFAULT_SOFT_FLOOR, MAX_SOFT_FLOOR)
    }

    /// Map a raw cosine ([-1,1]) into the soft-score space [0,1] used by the
    /// rolling/adaptive machinery: linear from `soft_floor()` → 0 to 1.0 → 1.
    #[must_use]
    pub(crate) fn soft_score(&self, cosine: f32) -> f32 {
        let floor = self.soft_floor();
        ((cosine - floor) / (1.0 - floor)).clamp(0.0, 1.0)
    }
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            neg_mean: 0.0,
            neg_std: 0.0,
            neg_p99: DEFAULT_SOFT_FLOOR,
            n_negatives: 0,
        }
    }
}

/// Persisted wake-word enrollment (schema v2): prototype + calibration.
///
/// The prototype is the L2-normalized mean of the accepted per-utterance
/// window embeddings.  Scoring is cosine similarity against the prototype,
/// mapped through [`Calibration::soft_score`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WakeWordEnrollment {
    #[serde(default)]
    pub(crate) schema_version: u32,
    /// Normalized wake word phrase (lowercased, trimmed).
    pub(crate) phrase: String,
    /// Expected embedding dimension (runtime: [`WAKE_WORD_EMBEDDING_DIM`]).
    pub(crate) embedding_dim: usize,
    /// L2-normalized prototype embedding (1024-dim).
    pub(crate) prototype: Vec<f32>,
    /// Number of utterances that went into the prototype.
    pub(crate) utterance_count: usize,
    /// Negative-sample calibration stats.
    #[serde(default)]
    pub(crate) calibration: Calibration,
    /// Anti-prototypes: up to [`MAX_NEGATIVE_PROTOTYPES`] L2-normalized
    /// cluster centroids of the negative-sample embeddings.
    ///
    /// Scoring hard-rejects a window that is closer to ANY anti-prototype
    /// than to the wake-word prototype (see [`WakeWordEnrollment::soft_score`]),
    /// so a window that resembles the enrolled wake word AND a calibration
    /// negative (e.g. a confusable near-miss like "madbot" vs "mahbot") is
    /// suppressed — the discriminative margin between near-identical phrases
    /// lives in which prototype is closer, not in the absolute cosine.  This
    /// is the no-trainable-head realization of negative-sample calibration.
    #[serde(default)]
    pub(crate) negative_prototypes: Vec<Vec<f32>>,
    /// RFC 3339 timestamp of initial creation (never updated).
    #[serde(default)]
    pub(crate) created_at: String,
    /// RFC 3339 timestamp of most recent training/persist operation.
    #[serde(default)]
    pub(crate) trained_at: String,
}

impl WakeWordEnrollment {
    /// Build a new enrollment from per-utterance window embeddings and the
    /// negative-sample pool.
    ///
    /// `utterance_embeddings` must already be L2-normalized (output of
    /// [`encode_window`]).  The prototype is their L2-normalized mean.
    /// `negative_embeddings` (also L2-normalized) are distilled into up to
    /// [`MAX_NEGATIVE_PROTOTYPES`] anti-prototypes via farthest-point
    /// sampling — the discriminative negative-sample calibration used by
    /// [`soft_score`](Self::soft_score).
    /// Returns `None` when fewer than [`MIN_ENROLLMENT_UTTERANCES`] are given.
    #[must_use]
    pub(crate) fn build(
        phrase: String,
        utterance_embeddings: &[Vec<f32>],
        calibration: Calibration,
        negative_embeddings: &[Vec<f32>],
        created_at: String,
        trained_at: String,
    ) -> Option<Self> {
        if utterance_embeddings.len() < MIN_ENROLLMENT_UTTERANCES {
            return None;
        }
        let dim = WAKE_WORD_EMBEDDING_DIM;
        let mut prototype = vec![0.0f32; dim];
        for emb in utterance_embeddings {
            if emb.len() != dim {
                return None;
            }
            for (p, e) in prototype.iter_mut().zip(emb) {
                *p += e;
            }
        }
        #[expect(clippy::cast_precision_loss)]
        let inv = 1.0 / utterance_embeddings.len() as f32;
        for p in &mut prototype {
            *p *= inv;
        }
        l2_normalize_in_place(&mut prototype);
        Some(Self {
            schema_version: ENROLLMENT_SCHEMA_VERSION,
            phrase,
            embedding_dim: dim,
            prototype,
            utterance_count: utterance_embeddings.len(),
            calibration,
            negative_prototypes: distill_negative_prototypes(negative_embeddings),
            created_at,
            trained_at,
        })
    }

    /// Cosine similarity of a window embedding (L2-normalized) against the
    /// prototype.
    #[must_use]
    pub(crate) fn cosine(&self, embedding: &[f32]) -> f32 {
        cosine_similarity(embedding, &self.prototype)
    }

    /// Maximum cosine of a window embedding against the anti-prototypes
    /// (0.0 when no negatives were collected).
    #[must_use]
    pub(crate) fn max_negative_cosine(&self, embedding: &[f32]) -> f32 {
        self.negative_prototypes
            .iter()
            .map(|a| cosine_similarity(embedding, a))
            .fold(0.0_f32, f32::max)
    }

    /// Soft score in [0,1] for a window embedding.
    ///
    /// Discriminative scoring with the negative-sample anti-prototypes:
    /// a window that is closer to ANY anti-prototype than to the wake-word
    /// prototype is rejected outright (score 0) — the confusable near-miss
    /// "madbot" resembles the enrolled "mahbot" prototype AND the "madbot"
    /// anti-prototype, but it is closer to the latter, so the comparison
    /// collapses it.  Windows closer to the prototype are scored by the
    /// plain positive cosine through the calibration floor.
    #[must_use]
    pub(crate) fn soft_score(&self, embedding: &[f32]) -> f32 {
        let pos = self.cosine(embedding);
        let neg = self.max_negative_cosine(embedding);
        if !self.negative_prototypes.is_empty() && neg > pos {
            // Closer to a negative prototype than to the wake word — reject.
            return 0.0;
        }
        self.calibration.soft_score(pos)
    }
}

/// Maximum number of anti-prototype centroids distilled from the negative
/// pool at enrollment (farthest-point sampling keeps the set spread).
const MAX_NEGATIVE_PROTOTYPES: usize = 8;

/// Distill the negative-sample pool into up to [`MAX_NEGATIVE_PROTOTYPES`]
/// L2-normalized anti-prototypes via farthest-point sampling.
///
/// Farthest-point (max-min) sampling picks the negative that is least similar
/// to the already-chosen anti-prototypes each round, so the small set covers
/// the full negative manifold (owner speech, ambient, confusables, unrelated)
/// rather than clustering on the densest region.  With ≤1 negative, returns
/// that single prototype; with none, returns an empty set (scoring falls back
/// to the plain positive cosine).
#[must_use]
pub(crate) fn distill_negative_prototypes(negatives: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if negatives.is_empty() {
        return Vec::new();
    }
    if negatives.len() == 1 {
        return vec![negatives[0].clone()];
    }
    let k = negatives.len().min(MAX_NEGATIVE_PROTOTYPES);
    let mut chosen: Vec<usize> = Vec::with_capacity(k);
    // Seed with the negative farthest from the origin-ish first candidate:
    // simply start with index 0 (deterministic).
    chosen.push(0);
    while chosen.len() < k {
        // For each unchosen negative, its distance to the chosen set is the
        // MINIMUM cosine similarity to any chosen prototype.  Pick the one
        // with the smallest such similarity (farthest from all chosen).
        let mut best_idx = None;
        let mut best_sim = f32::MAX;
        for (i, n) in negatives.iter().enumerate() {
            if chosen.contains(&i) {
                continue;
            }
            let min_sim = chosen
                .iter()
                .map(|&c| cosine_similarity(n, &negatives[c]))
                .fold(f32::MAX, f32::min);
            if min_sim < best_sim {
                best_sim = min_sim;
                best_idx = Some(i);
            }
        }
        match best_idx {
            Some(i) => chosen.push(i),
            None => break,
        }
    }
    chosen.into_iter().map(|i| negatives[i].clone()).collect()
}

/// Compute calibration stats from negative window embeddings (already
/// L2-normalized) against the prototype.
///
/// The floor is the **99th percentile** of the negative cosine distribution —
/// the confusable near-miss tail ("madbot" vs "mahbot") is the discriminative
/// signal (a p95 floor would leave ~5% of that tail above it).  With fewer
/// than 4 negatives, falls back to `mean + 2σ` (or the default floor when σ
/// is degenerate).
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub(crate) fn calibrate_negatives(prototype: &[f32], negatives: &[Vec<f32>]) -> Calibration {
    if negatives.is_empty() {
        return Calibration::default();
    }
    let mut cosines: Vec<f32> = negatives
        .iter()
        .map(|n| cosine_similarity(n, prototype))
        .collect();
    cosines.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = cosines.len();
    let mean = cosines.iter().sum::<f32>() / n as f32;
    let var = cosines.iter().map(|c| (c - mean) * (c - mean)).sum::<f32>() / n as f32;
    let std = var.sqrt();
    let p99 = if n >= 4 {
        let idx = ((n - 1) as f32 * 0.99).round() as usize;
        cosines[idx.min(n - 1)]
    } else {
        (mean + 2.0 * std).min(cosines[n - 1])
    };
    Calibration {
        neg_mean: mean,
        neg_std: std,
        // The floor is clamped into the meaningful soft-score range
        // [DEFAULT_SOFT_FLOOR, MAX_SOFT_FLOOR] — [`Calibration::soft_floor`]
        // re-clamps to the same range (the clamp here keeps the persisted
        // p99 field itself canonical).
        neg_p99: p99.clamp(DEFAULT_SOFT_FLOOR, MAX_SOFT_FLOOR),
        n_negatives: n,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(v: &[f32]) -> Vec<f32> {
        l2_normalize(v)
    }

    /// Deterministic unit vector whose cosine against `reference` is `c`.
    ///
    /// The orthogonal component is derived from a seeded random direction, so
    /// different `seed` values yield genuinely distinct directions — not
    /// scaled copies of the reference (a scaled+renormalized reference is
    /// directionally identical to it, which made the original calibration
    /// test vacuous).
    fn unit_with_cosine(reference: &[f32], c: f32, seed: u64) -> Vec<f32> {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut r: Vec<f32> = (0..reference.len())
            .map(|_| rng.random::<f32>() * 2.0 - 1.0)
            .collect();
        // Gram-Schmidt: orthogonalize r against the reference.
        let proj: f32 = reference.iter().zip(&r).map(|(a, b)| a * b).sum();
        for (ri, &ri_ref) in r.iter_mut().zip(reference) {
            *ri -= proj * ri_ref;
        }
        let ortho = l2_normalize(&r);
        let s = (1.0 - c * c).max(0.0).sqrt();
        let mut v: Vec<f32> = reference
            .iter()
            .zip(&ortho)
            .map(|(&a, &b)| c * a + s * b)
            .collect();
        l2_normalize_in_place(&mut v);
        v
    }

    #[test]
    fn calibration_floor_maps_negatives_to_zero() {
        let proto = norm(
            &(0..WAKE_WORD_EMBEDDING_DIM)
                .map(|i| i as f32)
                .collect::<Vec<_>>(),
        );

        // 20 genuinely distinct negative directions at cosines 0.20..0.44 —
        // all below the default floor.  p99 ≈ 0.44 clamps to the default
        // floor (0.55), and every negative maps to soft score 0.
        let negatives: Vec<Vec<f32>> = (0..20)
            .map(|i| unit_with_cosine(&proto, 0.20 + i as f32 * 0.0125, 1000 + i as u64))
            .collect();
        // Sanity: the negatives really sit at their claimed cosines with
        // distinct directions (guards the helper against collapsing onto the
        // prototype — the vacuous-pattern regression).
        for (i, n) in negatives.iter().enumerate() {
            let c = cosine_similarity(n, &proto);
            let expected = 0.20 + i as f32 * 0.0125;
            assert!(
                (c - expected).abs() < 1e-2,
                "negative {i} cosine {c} != expected {expected}"
            );
        }
        let cal = calibrate_negatives(&proto, &negatives);
        assert!(
            (cal.neg_p99 - DEFAULT_SOFT_FLOOR).abs() < 1e-6,
            "p99 below the floor must clamp to the default floor, got {}",
            cal.neg_p99,
        );
        assert_eq!(cal.soft_floor(), DEFAULT_SOFT_FLOOR);
        for n in &negatives {
            assert_eq!(cal.soft_score(cosine_similarity(n, &proto)), 0.0);
        }
        // A strong match (cosine ~0.9) maps well above 0.5.
        let match_emb = unit_with_cosine(&proto, 0.9, 999);
        assert!(cal.soft_score(cosine_similarity(&match_emb, &proto)) > 0.5);

        // Second band: negatives clustered HIGH (cosine ~0.70) — the p99
        // floor must follow the data (floor > default), proving calibration
        // is not just the default clamp.
        let hot_negatives: Vec<Vec<f32>> = (0..10)
            .map(|i| unit_with_cosine(&proto, 0.69 + i as f32 * 0.002, 2000 + i as u64))
            .collect();
        let hot = calibrate_negatives(&proto, &hot_negatives);
        assert!(
            hot.soft_floor() > DEFAULT_SOFT_FLOOR + 0.1,
            "floor must track the high negative band, got {}",
            hot.soft_floor(),
        );
        for n in &hot_negatives {
            assert_eq!(hot.soft_score(cosine_similarity(n, &proto)), 0.0);
        }
        let strong = unit_with_cosine(&proto, 0.9, 998);
        assert!(hot.soft_score(cosine_similarity(&strong, &proto)) > 0.5);
    }

    #[test]
    fn anti_prototype_gate_rejects_windows_closer_to_a_negative() {
        let proto = norm(
            &(0..WAKE_WORD_EMBEDDING_DIM)
                .map(|i| i as f32)
                .collect::<Vec<_>>(),
        );
        // A confusable: a distinct direction at cosine 0.85 vs the prototype —
        // close to the wake word, and it IS the negative prototype.
        let confusable = unit_with_cosine(&proto, 0.85, 42);
        let cal = Calibration::default(); // floor 0.55 → cosine 0.85 → soft 0.67
        assert!(cal.soft_score(cosine_similarity(&confusable, &proto)) > 0.5);

        let five: Vec<Vec<f32>> = vec![proto.clone(); MIN_ENROLLMENT_UTTERANCES];
        let enr = WakeWordEnrollment::build(
            "mahbot".into(),
            &five,
            cal.clone(),
            &[confusable.clone()],
            String::new(),
            String::new(),
        )
        .expect("enrollment");

        // The confusable is closer to the anti-prototype (cosine 1.0) than to
        // the prototype (cosine 0.85) → hard-rejected to 0.
        assert_eq!(enr.soft_score(&confusable), 0.0);
        // The exact prototype embedding is closer to the prototype (1.0) than
        // to the anti-prototype (0.85) → scored normally.
        assert!(enr.soft_score(&proto) > 0.5);
        // Without anti-prototypes the same confusable is NOT rejected — the
        // gate (not the floor) is what suppressed it.
        let no_neg = WakeWordEnrollment::build(
            "mahbot".into(),
            &five,
            cal.clone(),
            &[],
            String::new(),
            String::new(),
        )
        .expect("enrollment");
        assert!(no_neg.soft_score(&confusable) > 0.5);
    }

    #[test]
    fn distill_negative_prototypes_spreads_selection() {
        let proto = norm(
            &(0..WAKE_WORD_EMBEDDING_DIM)
                .map(|i| i as f32)
                .collect::<Vec<_>>(),
        );
        // 24 well-separated negative directions (mutually near-orthogonal
        // via distinct seeds; cosine vs proto 0.30..0.53).
        let negatives: Vec<Vec<f32>> = (0..24)
            .map(|i| unit_with_cosine(&proto, 0.30 + i as f32 * 0.01, 3000 + i as u64))
            .collect();
        let distilled = distill_negative_prototypes(&negatives);
        assert!(!distilled.is_empty());
        assert!(distilled.len() <= MAX_NEGATIVE_PROTOTYPES);
        assert!(
            distilled.len() >= 2,
            "24 spread negatives should yield >1 prototype, got {}",
            distilled.len(),
        );
        // The distilled set must be genuinely spread: the maximum pairwise
        // cosine inside the set is well below 1.0 (distinct directions, not
        // a clustered re-pick).
        let mut max_pairwise = 0.0f32;
        for (i, a) in distilled.iter().enumerate() {
            for b in distilled.iter().skip(i + 1) {
                max_pairwise = max_pairwise.max(cosine_similarity(a, b));
            }
        }
        assert!(
            max_pairwise < 0.9,
            "distilled prototypes should be spread, max pairwise cosine {max_pairwise}",
        );
        // The distilled set covers the input: each chosen prototype is one of
        // the original negatives (L2-normalized).
        for d in &distilled {
            assert!(
                negatives
                    .iter()
                    .any(|n| (cosine_similarity(n, d) - 1.0).abs() < 1e-4)
            );
        }

        // Edge cases: single negative → exactly that prototype; empty → empty.
        let single = distill_negative_prototypes(&negatives[..1]);
        assert_eq!(single.len(), 1);
        assert!((cosine_similarity(&single[0], &negatives[0]) - 1.0).abs() < 1e-4);
        assert!(distill_negative_prototypes(&[]).is_empty());
    }

    #[test]
    fn calibration_default_floor() {
        let cal = Calibration::default();
        assert_eq!(cal.soft_floor(), DEFAULT_SOFT_FLOOR);
        assert_eq!(cal.soft_score(DEFAULT_SOFT_FLOOR), 0.0);
        assert_eq!(cal.soft_score(1.0), 1.0);
    }

    #[test]
    fn build_requires_min_utterances() {
        let emb = norm(&vec![1.0; WAKE_WORD_EMBEDDING_DIM]);
        let enough: Vec<Vec<f32>> = vec![emb.clone(); MIN_ENROLLMENT_UTTERANCES];
        assert!(
            WakeWordEnrollment::build(
                "mahbot".into(),
                &enough,
                Calibration::default(),
                &[],
                String::new(),
                String::new(),
            )
            .is_some()
        );
        let too_few: Vec<Vec<f32>> = vec![emb; MIN_ENROLLMENT_UTTERANCES - 1];
        assert!(
            WakeWordEnrollment::build(
                "mahbot".into(),
                &too_few,
                Calibration::default(),
                &[],
                String::new(),
                String::new(),
            )
            .is_none()
        );
    }

    #[test]
    fn prototype_is_unit_norm() {
        let emb = norm(
            &(0..WAKE_WORD_EMBEDDING_DIM)
                .map(|i| i as f32)
                .collect::<Vec<_>>(),
        );
        let five: Vec<Vec<f32>> = vec![emb; MIN_ENROLLMENT_UTTERANCES];
        let enr = WakeWordEnrollment::build(
            "mahbot".into(),
            &five,
            Calibration::default(),
            &[],
            String::new(),
            String::new(),
        )
        .expect("enrollment");
        let norm: f32 = enr.prototype.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3);
        assert_eq!(enr.schema_version, ENROLLMENT_SCHEMA_VERSION);
    }

    #[test]
    fn serde_roundtrip_v2() {
        let emb = norm(
            &(0..WAKE_WORD_EMBEDDING_DIM)
                .map(|i| i as f32)
                .collect::<Vec<_>>(),
        );
        let eight: Vec<Vec<f32>> = vec![emb; 8];
        let enr = WakeWordEnrollment::build(
            "mahbot".into(),
            &eight,
            Calibration::default(),
            &[],
            "created".into(),
            "trained".into(),
        )
        .expect("enrollment");
        let json = serde_json::to_string(&enr).expect("serialize");
        let back: WakeWordEnrollment = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.schema_version, ENROLLMENT_SCHEMA_VERSION);
        assert_eq!(back.embedding_dim, WAKE_WORD_EMBEDDING_DIM);
        assert_eq!(back.prototype.len(), WAKE_WORD_EMBEDDING_DIM);
        assert_eq!(back.phrase, "mahbot");
    }

    #[test]
    fn v1_schema_rejected() {
        // A v1 record (classifier-based) must not deserialize as v2.
        let v1 = r#"{"schema_version":1,"phrase":"mahbot","embedding_dim":96,
                     "window_size":3,"classifier":[]}"#;
        let parsed: serde_json::Result<WakeWordEnrollment> = serde_json::from_str(v1);
        // v1 has no prototype → deserialization fails (missing field).
        assert!(parsed.is_err());
    }
}
