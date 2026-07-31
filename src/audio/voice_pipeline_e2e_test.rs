//! E2E integration test / benchmark for the full voice pipeline (mahbot-811, mahbot-844).
//!
//! This test exercises the **enrollment-to-detection cycle with realistic
//! TTS-generated speech audio**.  It uses the TTS engine to synthesize wake
//! word variants, feeds them through the enrollment pipeline, trains the
//! Conv1D classifier and VoiceVerifier, then runs detection on:
//!
//! * Positive cases (wake word variants)
//! * Negative — confusable near-miss phrases
//! * Negative — completely unrelated speech
//! * Negative — silence and noise
//!
//! # Running as benchmark (recommended)
//!
//! ```sh
//! cargo bench --bench voice_pipeline_e2e
//! ```
//!
//! The benchmark uses `[profile.bench]` (opt-level=3, fat LTO, codegen-units=1)
//! for maximum performance.  First run populates the TTS audio cache
//! (~14-17 min); subsequent runs complete in ~2-3 min.
//!
//! # Requirements
//!
//! * TTS models must be downloaded and cached (run the app once).
//! * Voice ONNX models (melspectrogram + embedding) must be present in
//!   `~/.mahbot/models/openwakeword/`.

// Clippy allowances — use super::* is intentional (mirrors the voice module's API surface).
// Cast warnings: this is a benchmark file with known-safe numeric conversions.
#![allow(
    clippy::wildcard_imports,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

use super::*; // voice module items (process_enrollment_sample, etc.)
use crate::audio::embedding_sequence::{EmbeddingSequence, Source, UtteranceId};
use crate::audio::tts;
use crate::audio::voice_verifier::{
    CALIBRATION_LAMBDA, CALIBRATION_SWEEP_STEP, CONV_L2_LAMBDA, DEFAULT_VERIFIER_THRESHOLD,
    VoiceVerifier,
};
use crate::audio::wake_word_classifier::WakeWordClassifier;
use earshot::Detector;
use rand::{RngExt, SeedableRng};
use std::borrow::Cow;
use std::time::Instant;

// ── Constants ──────────────────────────────────────────────────────────────

/// Index of the rolling sum field in per_frame_scores `[total_score, rolling_sum, threshold]` triples.
const ROLLING_SUM_IDX: usize = 1;
/// Index of the effective threshold field in per_frame_scores `[total_score, rolling_sum, threshold]` triples.
const THRESHOLD_IDX: usize = 2;

/// Minimum effective threshold for the Conv1D classifier's rolling sum.
///
/// The classifier rolling sum must reach at least this value for the
/// classifier to have fired.  Below this threshold, misses are attributed
/// to the classifier (mahbot-882).
///
/// Updated for mahbot-923's unified dense stride-8 embeddings (1.58× multiplier
/// vs the old mixed old-style + streaming distribution): 2.13 = 1.35 * 1.58.
const MIN_CLASSIFIER_THRESHOLD: f32 = 2.13;

/// Number of samples to prepend for verifier warm-up (mahbot-922, mahbot-926).
///
/// ~1.28 s of noise at 16 kHz, consumed by the verifier warm-up period
/// (4 embeddings, see [`VERIFIER_WARMUP_EMBEDDINGS`](crate::audio::voice_verifier::VERIFIER_WARMUP_EMBEDDINGS))
/// so the actual test utterance arrives with a fully active verifier — matching
/// production behaviour where the warm-up is consumed during background
/// silence/noise before anyone speaks.
///
/// ## Derivation (mahbot-926, stride-8 alignment)
///
/// After mahbot-923 switched to stride-8 sliding-window embedding extraction,
/// each embedding requires [`EMBEDDING_WINDOW_FRAMES`] (76) consecutive mel
/// frames.  Mel frames are produced by the ONNX `melspectrogram` model at
/// 10 ms intervals (hop = 160 samples at 16 kHz).
///
/// The number of mel frames from N audio samples:
/// ```text
/// mel_frames = max(0, (N - 512) / 160 + 1)
/// ```
/// where 512 is the STFT window size and 160 is the hop length.
///
/// The stride-8 sliding window produces one embedding every 8 mel frames
/// (80 ms).  To get `N` embeddings we need at least `76 + (N-1) × 8` mel
/// frames.  For `N = VERIFIER_WARMUP_EMBEDDINGS = 4`:
/// ```text
/// mel_frames_needed = 76 + 3 × 8 = 100
/// samples_needed   ≈ (100 - 1) × 160 + 512 = 16352
/// ```
///
/// 20480 samples (~1.28 s, ~125 mel frames) produces 7 stride-8 embeddings,
/// safely above the minimum of 4.  The excess margin (~3 extra embeddings)
/// provides robustness against pipeline changes (stride, window size, VAD
/// sensitivity).  If the embedding window or stride changes in `voice.rs`,
/// this constant should be recalculated to maintain ≥ 4 embeddings.
const WARMUP_PREPEND_SAMPLES: usize = 20480; // 1.28s × 16 kHz

/// TTS phrase for verifier warm-up audio (mahbot-947).
///
/// A short non-wake-word phrase synthesised via the already-loaded TTS engine.
/// Speech-like harmonics guarantee the Earshot neural VAD triggers, producing
/// enough embeddings to consume the verifier warm-up period before the actual
/// test utterance is processed.
///
/// Must NOT contain the wake word ("hey mahbot") or phonetically similar
/// phrases that could trigger the Conv1D classifier.
const WARMUP_TTS_PHRASE: &str = "testing one two three";

/// Cached TTS warm-up audio, populated on first successful synthesis.
/// Unlike the original `WARMUP_NOISE_CACHE`, this caches ONLY the TTS
/// result — if TTS is unavailable on the first call, a fresh pink-noise
/// fallback is returned (no caching), so TTS is re-evaluated on subsequent
/// calls (mahbot-947, reviewer feedback on cache poisoning).
static WARMUP_TTS_CACHE: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();

/// Default wake word for the test.
const WAKE_WORD: &str = "hey mahbot";

/// Number of enrollment variants to generate (mahbot-932 Fix 5).
///
/// Production generates ~45-50 positive training sequences from 10 utterances
/// × 5 PCM variants (minus SpeedUp skip for short utterances).  We target
/// ~10 TTS enrollment variants to match this.
const NUM_ENROLLMENT_VARIANTS: usize = 10;

/// Owner-negative (non-wake-word) phrases for verifier/classifier training
/// (mahbot-932 Fix 6).
///
/// These are generated via TTS, tagged as `Source::Owner`, and used to help
/// the models reject general speech from the enrolled user.  Documented
/// limitation: TTS speech cannot match the distribution of real human Phase 3
/// speech collected during production enrollment.
const OWNER_NEGATIVE_PHRASES: &[&str] = &[
    "hello",
    "good morning",
    "turn on the lights",
    "what time is it",
    "thank you",
];

/// Minimum detection rate for positive (wake word) variants required to pass.
///
/// NOTE (mahbot-997): The benchmark now auto-computes a suggested value for
/// this constant via the post-hoc verifier threshold analysis.  To update:
///
/// 1. Run the benchmark 3× to observe verifier variance — the verifier seed is
///    `None` (entropy-based RNG) since mahbot-1006 K, matching production.
/// 2. Take the mean of the three `computed_min_dr` values from the report.
/// 3. Update this constant to `floor(mean / 0.2) * 0.2`.
///
/// The 0.2 granularity reflects the ~17-variant test set (after PCM
/// augmentation × 10 enrollment variants, split 2/3:1/3) where each miss
/// costs roughly 6 percentage points.
const MIN_DETECTION_RATE: f64 = 0.60;

/// Minimum verifier recall on positive (wake-word) test variants (mahbot-1008
/// Fix 6).
///
/// The verifier must accept at least this fraction of genuine wake-word
/// variants that it actually evaluates (verifier trained + warm-up complete +
/// classifier crossed the effective threshold).  The pre-fix verifier scored
/// `6.67e-8` on every out-of-distribution input — a 0% accept rate — while the
/// benchmark reported "0% detection" as a success because it had no
/// verifier-recall metric.
///
/// ## Report-only semantics (mahbot-953 precedent)
///
/// This constant gates a prominent WARNING, not a hard assert: mahbot-953
/// deliberately removed all benchmark pass/fail gating, and the mahbot-997
/// auto-suggestion protocol can ratchet `MIN_DETECTION_RATE` toward 0.0 on
/// failing runs (see the detection-rate hint below, which now refuses to
/// endorse lowering the constant on a failing run).  A hard gate can be
/// re-introduced in a follow-up ticket once fixes #1-#5 have been validated
/// across benchmark runs.
const VERIFIER_RECALL_MIN: f64 = 0.90;

// Per-category false accept limits are now dynamic by tier — see
// [`tier_limits`] and [`BenchTier`] (mahbot-871).

/// Confusable near-miss phrases for negative detection testing (mahbot-834).
///
/// Uses the canonical list from the parent `voice` module (mahbot-859).
const CONFUSABLE_PHRASES: &[&str] = super::CONFUSABLE_PHRASES;
const CONFUSABLE_HARD: &[&str] = super::CONFUSABLE_HARD;
const CONFUSABLE_MEDIUM: &[&str] = super::CONFUSABLE_MEDIUM;
const CONFUSABLE_EASY: &[&str] = super::CONFUSABLE_EASY;

/// Unrelated speech phrases for negative detection testing (mahbot-834).
///
/// Uses the canonical list from the parent `voice` module (mahbot-872).
const UNRELATED_PHRASES: &[&str] = super::UNRELATED_PHRASES;

/// Silence audio length in samples (1 second at 16 kHz).
const SILENCE_LEN: usize = 16_000;

/// Noise audio length in samples (1 second at 16 kHz).
const NOISE_LEN: usize = 16_000;

/// Noise profiles for negative detection testing (mahbot-834).
///
/// Each noise profile is a (label, generator_fn) pair.  The generator
/// produces PCM f32 samples at 16 kHz.
type NoiseGenerator = fn() -> Vec<f32>;

const NOISE_PROFILES: &[(&str, NoiseGenerator)] = &[
    ("white uniform noise", generate_white_uniform_noise),
    ("white gaussian noise", generate_white_gaussian_noise),
    ("pink noise", generate_pink_noise),
    ("brown noise", generate_brown_noise),
    ("mixed speech+noise", generate_mixed_speech_noise),
    ("blue noise", generate_blue_noise),
    ("violet noise", generate_violet_noise),
    ("low-frequency rumble", generate_low_freq_rumble),
    ("modulated noise", generate_modulated_noise),
    ("high-frequency hiss", generate_high_freq_hiss),
];

/// TTS target sample rate (voice pipeline rate).
const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Default TTS voice style when no styles are available from disk.
/// This matches the naming convention used by the TTS model download.
const DEFAULT_TTS_STYLE: &str = "M1.json";

/// Cache directory relative to storage root.
const TEST_CACHE_DIR: &str = "test_cache/voice_e2e";

/// SNR levels for the noise-overlapped detection test (mahbot-845).
/// Each entry is (label, snr_db).  Clean = infinity dB (no noise added).
const NOISE_OVERLAP_SNRS: &[(&str, f32)] = &[
    ("clean", f32::INFINITY),
    ("20dB", 20.0),
    ("10dB", 10.0),
    ("5dB", 5.0),
    ("0dB", 0.0),
];

/// Noise types to use for noise-overlapped detection tests.
/// White (uniform), Pink, and Brown noise (mahbot-845).
const NOISE_OVERLAP_TYPES: &[(&str, NoiseGenerator)] = &[
    ("white", generate_white_uniform_noise),
    ("pink", generate_pink_noise),
    ("brown", generate_brown_noise),
];

// ── Tiered benchmark configuration (mahbot-871) ────────────────────────────

/// Difficulty tiers for confusable phrase testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchTier {
    Easy,
    Medium,
    Hard,
}

impl BenchTier {
    /// Index into a 3-element array: Easy→0, Medium→1, Hard→2.
    const fn index(self) -> usize {
        match self {
            BenchTier::Easy => 0,
            BenchTier::Medium => 1,
            BenchTier::Hard => 2,
        }
    }

    /// Stable string label for JSON and human-readable output.
    const fn as_str(self) -> &'static str {
        match self {
            BenchTier::Easy => "easy",
            BenchTier::Medium => "medium",
            BenchTier::Hard => "hard",
        }
    }
}

/// Per-category false-accept limits for a given tier.
#[derive(Debug, Clone, Copy)]
struct TierLimits {
    confusable: usize,
    noise: usize,
    total: usize,
}

/// Return the per-category false-accept limits for the given tier.
const fn tier_limits(tier: BenchTier) -> TierLimits {
    match tier {
        BenchTier::Easy => TierLimits {
            confusable: 0,
            noise: 0,
            total: 0,
        },
        BenchTier::Medium => TierLimits {
            confusable: 1,
            noise: 1,
            total: 1, // combined confusable+noise cap (tighter than sum of individual limits)
        },
        BenchTier::Hard => TierLimits {
            confusable: 1,
            noise: 1,
            total: 2,
        },
    }
}

/// Extract the phrase text from a generate_phrase_variants_cached label.
///
/// Label format: `"{prefix}_{phrase}_s{i}"` e.g. `"confusable_hey madbot_s0"`.
///
/// The seed suffix `_s{i}` (underscore + 's' + digits) is always the
/// rightmost `_s` in the label, so `rfind` finds it correctly even if the
/// phrase text itself contains the substring `_s`. We additionally verify
/// that the `_s` is followed by at least one ASCII digit as a sanity check
/// against label-format drift.
fn phrase_from_label<'a>(label: &'a str, prefix: &str) -> &'a str {
    let after_prefix = label
        .strip_prefix(prefix)
        .and_then(|r| r.strip_prefix('_'))
        .unwrap_or(label);
    // The seed suffix `_s{i}` is always the rightmost `_s` in the label.
    if let Some(idx) = after_prefix.rfind("_s") {
        let after_s = &after_prefix[idx + 2..];
        if after_s.starts_with(|c: char| c.is_ascii_digit()) {
            return &after_prefix[..idx];
        }
    }
    // No valid seed suffix found; return the full string after the prefix.
    after_prefix
}

/// Determine the difficulty tier for a confusable phrase.
fn tier_for_phrase(phrase: &str) -> BenchTier {
    if CONFUSABLE_HARD.contains(&phrase) {
        BenchTier::Hard
    } else if CONFUSABLE_MEDIUM.contains(&phrase) {
        BenchTier::Medium
    } else if CONFUSABLE_EASY.contains(&phrase) {
        BenchTier::Easy
    } else {
        unreachable!(
            "Confusable phrase '{phrase}' not found in any tier array. \
             The phrase must exist in exactly one of CONFUSABLE_HARD/MEDIUM/EASY, \
             and those arrays must together cover all entries in CONFUSABLE_PHRASES."
        )
    }
}

// ── TTS audio cache (mahbot-844 Part 1) ─────────────────────────────────────
/// clipping after boost) — so the gain parameter is 12.0 with clip=true.
const VOLUME_SWEEP_LEVELS: &[(&str, f32, bool)] = &[
    ("minus_12dB", -12.0, false),
    ("baseline_0dB", 0.0, false),
    ("plus_6dB", 6.0, false),
    ("plus_12dB", 12.0, false),
    ("hard_clipped", 12.0, true), // +12dB gain then hard-clamp at ±1.0
];

// ── TTS audio cache (mahbot-844 Part 1) ─────────────────────────────────────

// Compute a deterministic model version hash from TTS compile-time SHA-256
// constants (mahbot-844 Part 1).
//
// Hashes the concatenation of all TTS model SHA-256 digests and the voice
// style directory hash.  These constants are updated in `tts.rs` whenever
// model files change, so the cache key automatically tracks model version
// without disk I/O.
//
// Returns a hex string (always succeeds since the hashes are compile-time).
//
// Delegates to [`super::tts_model_version_hash`] which is the canonical
// implementation from `voice.rs`.

/// Get the cache directory path.
fn cache_dir() -> std::path::PathBuf {
    let root = crate::config::CONFIG
        .try_storage_root()
        .expect("CONFIG storage root must be set");
    root.join(TEST_CACHE_DIR)
}

/// Synthesize wake word variant audio with TTS caching support.
///
/// Delegates to the production implementation.
fn synthesize_wake_word_variant_cached(
    text: &str,
    style: &str,
    seed: u64,
    sample_rate: u32,
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Option<Vec<f32>> {
    super::synthesize_with_pcm_cache(text, style, seed, sample_rate, model_hash, cache_dir)
}

// ── Audio generation helpers (with cache) ─────────────────────────────────

/// Generate enrollment audio variants (different voices, seeds) using TTS
/// caching. Returns them as `(samples, label)` tuples.
fn generate_enrollment_variants_cached(
    available_styles: &[String],
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Vec<(Vec<f32>, String)> {
    let mut variants = Vec::new();
    let num_styles = available_styles.len();
    if num_styles == 0 {
        warn!("No TTS voice styles available — using default style");
        for i in 0..NUM_ENROLLMENT_VARIANTS {
            if let Some(pcm) = synthesize_wake_word_variant_cached(
                WAKE_WORD,
                DEFAULT_TTS_STYLE,
                i as u64 + 100,
                TARGET_SAMPLE_RATE,
                model_hash,
                cache_dir,
            ) {
                variants.push((pcm, format!("default_style_var{i}")));
            }
        }
    } else {
        for i in 0..NUM_ENROLLMENT_VARIANTS {
            let style = &available_styles[i % num_styles];
            let seed = i as u64 + 100;
            if let Some(pcm) = synthesize_wake_word_variant_cached(
                WAKE_WORD,
                style,
                seed,
                TARGET_SAMPLE_RATE,
                model_hash,
                cache_dir,
            ) {
                variants.push((pcm, format!("{style}_enroll{i}")));
            }
        }
    }
    variants
}

/// Seed configuration for TTS phrase variant generation.
///
/// Encapsulates the three related seed parameters that control deterministic
/// TTS synthesis of phrase variants with different seeds and style rotations.
/// Bundled into a struct for call-site clarity (mahbot-872 reviewer feedback).
#[derive(Clone, Copy)]
struct SeedConfig {
    /// Base seed for deterministic TTS synthesis of the phrase list.
    base_seed: u64,
    /// Number of seed variants — determines style rotation stride and seed
    /// spacing.  Must match the `num_variants` passed to the batch caller.
    num_variants: usize,
    /// Which variant index this call generates (0..num_variants).  Used in
    /// the style distribution and seed formulas.
    seed_variant: usize,
}

/// Generate TTS audio for a list of phrases with caching.
fn generate_phrase_variants_cached(
    phrases: &[&str],
    available_styles: &[String],
    seed: SeedConfig,
    prefix: &str,
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Vec<(Vec<f32>, String)> {
    let mut variants = Vec::new();
    let num_styles = available_styles.len().max(1);

    for (i, &phrase) in phrases.iter().enumerate() {
        // Use the same round-robin style distribution as production
        // (mahbot-872 per-phrase seed formula): rotate styles across
        // (phrase × seed_variant) combos.
        let style_idx = (i * seed.num_variants + seed.seed_variant) % num_styles;
        let style = if available_styles.is_empty() {
            DEFAULT_TTS_STYLE
        } else {
            &available_styles[style_idx]
        };
        // Production seed formula (mahbot-872):
        //   seed = base_seed + i * num_variants + seed_variant
        let seed_val =
            seed.base_seed + i as u64 * seed.num_variants as u64 + seed.seed_variant as u64;

        if let Some(pcm) = synthesize_wake_word_variant_cached(
            phrase,
            style,
            seed_val,
            TARGET_SAMPLE_RATE,
            model_hash,
            cache_dir,
        ) {
            variants.push((pcm, format!("{prefix}_{phrase}_s{i}")));
        }
    }

    variants
}

/// Generate all seed variants for a phrase list in a single batch.
/// Generate white uniform noise in [-1.0, 1.0].
fn generate_white_uniform_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    (0..NOISE_LEN)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect()
}

/// Generate white Gaussian noise (approximately) in [-1.0, 1.0].
/// Uses the Box-Muller transform on uniform samples.
fn generate_white_gaussian_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(43);
    let mut samples = Vec::with_capacity(NOISE_LEN);
    let mut i = 0;
    while i < NOISE_LEN {
        let u1: f32 = rng.random::<f32>().max(f32::EPSILON);
        let u2: f32 = rng.random::<f32>().max(f32::EPSILON);
        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f32::consts::PI * u2).cos();
        let z2 = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f32::consts::PI * u2).sin();
        // Clamp to [-1.0, 1.0] — Gaussian has tails beyond [-3, 3] but
        // scaling by 0.333 keeps ~99.7% within [-1, 1].
        samples.push((z1 * 0.333).clamp(-1.0, 1.0));
        if i + 1 < NOISE_LEN {
            samples.push((z2 * 0.333).clamp(-1.0, 1.0));
        }
        i += 2;
    }
    samples
}

/// Generate pink noise (1/f spectrum) using the Voss-McCartney algorithm.
/// Produces approximately -3 dB/octave rolloff.
///
/// Uses the canonical [`crate::util::generate_pink_noise`] with a reproducible
/// seed (44) for deterministic benchmark output.
fn generate_pink_noise() -> Vec<f32> {
    crate::util::generate_pink_noise(NOISE_LEN, rand::rngs::StdRng::seed_from_u64(44))
}

/// Generate brown noise (integrated white noise, 1/f² spectrum).
/// Produces approximately -6 dB/octave rolloff — deeper, rumbling sound.
fn generate_brown_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(45);
    let mut samples = Vec::with_capacity(NOISE_LEN);
    let mut prev = 0.0;
    for _ in 0..NOISE_LEN {
        let white: f32 = rng.random::<f32>() * 2.0 - 1.0;
        // Leaky integrator to prevent DC drift
        prev = (prev + white * 0.125) * 0.98;
        samples.push(prev.clamp(-1.0, 1.0));
    }
    samples
}

/// Generate mixed speech+noise by overlapping a wake-word-like recording
/// with brown noise at low SNR (<5 dB) — simulating far-field / cocktail
/// party conditions where the wake word might acoustically resemble noise.
fn generate_mixed_speech_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(46);
    // Generate a tonal hum at ~200 Hz (close to male speech formant) mixed
    // with noise — simulating distant speech that might trigger VAD.
    let mut samples = Vec::with_capacity(NOISE_LEN);
    for i in 0..NOISE_LEN {
        let t = i as f32 / TARGET_SAMPLE_RATE as f32;
        let tone = (2.0 * core::f32::consts::PI * 200.0 * t).sin() * 0.15;
        let noise: f32 = rng.random::<f32>() * 2.0 - 1.0;
        // Low SNR: noise dominates, with tonal speech component
        samples.push((tone + noise * 0.85).clamp(-1.0, 1.0));
    }
    samples
}

/// Generate blue noise (spectral density increases 3dB/octave).
fn generate_blue_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(47);
    let mut prev: f32 = 0.0;
    (0..NOISE_LEN)
        .map(|_| {
            let white = rng.random::<f32>() * 2.0 - 1.0;
            // First-order difference approximates blue noise
            let blue = (white - prev) * 0.5;
            prev = white;
            blue.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Generate violet noise (spectral density increases 6dB/octave).
fn generate_violet_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(48);
    let mut prev1 = 0.0f32;
    let mut prev2 = 0.0f32;
    (0..NOISE_LEN)
        .map(|_| {
            let white = rng.random::<f32>() * 2.0 - 1.0;
            // Second-order difference approximates violet noise
            let violet = (white - 2.0 * prev1 + prev2) * 0.25;
            prev2 = prev1;
            prev1 = white;
            violet.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Generate low-frequency rumble (pink noise low-passed at ~300Hz).
fn generate_low_freq_rumble() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(49);
    let mut low: f32 = 0.0;
    let alpha = 0.95; // strong low-pass
    (0..NOISE_LEN)
        .map(|_| {
            let white = rng.random::<f32>() * 2.0 - 1.0;
            low = low * alpha + white * (1.0 - alpha);
            low.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Generate amplitude-modulated noise (tremolo effect).
fn generate_modulated_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(50);
    let mut samples = Vec::with_capacity(NOISE_LEN);
    for i in 0..NOISE_LEN {
        let t = i as f32 / TARGET_SAMPLE_RATE as f32;
        let modulator = (2.0 * core::f32::consts::PI * 4.0 * t).sin() * 0.5 + 0.5; // 4Hz tremolo
        let noise: f32 = rng.random::<f32>() * 2.0 - 1.0;
        samples.push((noise * modulator).clamp(-1.0, 1.0));
    }
    samples
}

/// Generate high-frequency hiss (high-pass filtered above ~4kHz).
fn generate_high_freq_hiss() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(51);
    let mut prev: f32 = 0.0;
    (0..NOISE_LEN)
        .map(|_| {
            let white = rng.random::<f32>() * 2.0 - 1.0;
            // Simple high-pass: y[n] = 0.9 * (x[n] - x[n-1]) + 0.8 * y[n-1]
            let hiss = 0.9 * (white - prev) + 0.8 * prev;
            prev = white;
            hiss.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Generate warm-up audio that reliably triggers the Earshot neural VAD.
///
/// Attempts TTS synthesis first (guaranteed speech-like harmonics), falling
/// back to pink noise + 200 Hz tone if TTS is unavailable (mahbot-947).
///
/// Returns a [`Cow`] that either borrows from the TTS cache (once populated)
/// or owns freshly-generated pink noise (not cached, so TTS is re-evaluated
/// on the next call if it becomes available — mahbot-947, reviewer feedback).
///
/// ## Determinism
/// TTS synthesis uses a fixed seed (947 = ticket number), producing the same
/// PCM output on every call (barring TTS model changes).  The result is cached
/// in [`WARMUP_TTS_CACHE`] so subsequent calls are zero-cost.
///
/// ## Fallback rationale
/// The pink noise + 200 Hz tone was the original warm-up signal (mahbot-922/926),
/// but its lack of speech-like harmonic structure causes the Earshot neural VAD
/// to reject it as non-speech, producing 0 embeddings and leaving the verifier
/// cold.  This was the root cause of Phase 8's 0% detection rate (mahbot-947).
fn generate_warmup_noise() -> Cow<'static, [f32]> {
    // TTS cache hit — fast path, no synthesis needed.
    if let Some(cached) = WARMUP_TTS_CACHE.get() {
        return Cow::Borrowed(cached);
    }

    // First call (or cache not yet populated): try TTS synthesis.
    // Speech-like harmonics guarantee Earshot VAD triggers, producing
    // enough mel frames for ≥7 stride-8 embeddings.
    if let Some(pcm) = try_warmup_tts() {
        let cached = WARMUP_TTS_CACHE.get_or_init(|| pcm);
        // Safety: we just verified get() returns None above; no race
        // because get_or_init is idempotent and returns the same &Vec.
        return Cow::Borrowed(cached);
    }

    // TTS unavailable — return fresh fallback (not cached, so we retry
    // TTS on future calls if models become available).
    warn!(
        "TTS warm-up synthesis failed — falling back to pink noise + tone. \
         This may not trigger Earshot VAD, causing 0 warm-up embeddings. \
         Ensure TTS models are cached (~/.mahbot/models/tts/)."
    );
    Cow::Owned(generate_warmup_noise_fallback())
}

/// Attempt TTS synthesis of the warm-up phrase.
///
/// Returns `None` if TTS is not initialised or synthesis fails, allowing the
/// caller to fall back to pink noise.  Logs an `info!` on success with the
/// output duration so maintainers can verify the warm-up length.
fn try_warmup_tts() -> Option<Vec<f32>> {
    let pcm = match crate::audio::tts::synthesize(
        WARMUP_TTS_PHRASE,
        DEFAULT_TTS_STYLE,
        947, // seed = ticket number for determinism
        TARGET_SAMPLE_RATE,
    ) {
        Ok(p) => p,
        Err(e) => {
            warn!("TTS warm-up synthesis failed: {e}");
            return None;
        }
    };

    if pcm.len() > WARMUP_PREPEND_SAMPLES {
        warn!(
            "Warm-up TTS output ({} samples = {:.2}s) exceeds \
             WARMUP_PREPEND_SAMPLES ({}) — truncating to {}",
            pcm.len(),
            pcm.len() as f64 / f64::from(TARGET_SAMPLE_RATE),
            WARMUP_PREPEND_SAMPLES,
            WARMUP_PREPEND_SAMPLES,
        );
        Some(pcm[..WARMUP_PREPEND_SAMPLES].to_vec())
    } else {
        info!(
            "Warm-up audio: TTS phrase '{}' ({:.2}s = {} samples) — VAD will trigger",
            WARMUP_TTS_PHRASE,
            pcm.len() as f64 / f64::from(TARGET_SAMPLE_RATE),
            pcm.len(),
        );
        Some(pcm)
    }
}

/// Generate the original pink noise + 200 Hz tone warm-up signal (mahbot-922).
///
/// Retained as a fallback when TTS models are not cached.  Properties:
/// 1. **VAD-inactive**: low probability of triggering the Earshot neural VAD
///    (no speech harmonics), which is the whole reason we now prefer TTS.
/// 2. **No false detection**: classifier scores are near-zero because pink
///    noise + 200 Hz tone does not resemble the "hey mahbot" embedding.
/// 3. **Deterministic**: RNG seed 922 (= original ticket number).
/// 4. **Aperiodic noise floor**: pink noise avoids periodic artefacts that
///    could bias the AGC steady state.
fn generate_warmup_noise_fallback() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(922);

    // Pink noise normalised to unit RMS at ~0.20 amplitude (≈ -14 dB).
    let pink = crate::util::generate_pink_noise(WARMUP_PREPEND_SAMPLES, &mut rng);
    let pink_gain = 0.20;

    // 200 Hz tonal component at ~0.10 amplitude (≈ -20 dB).
    let mut samples = Vec::with_capacity(WARMUP_PREPEND_SAMPLES);
    for (i, &p) in pink.iter().enumerate() {
        let t = i as f32 / TARGET_SAMPLE_RATE as f32;
        let tone = (2.0 * core::f32::consts::PI * 200.0 * t).sin() * 0.10;
        samples.push(p * pink_gain + tone);
    }

    samples
}

/// Process a single frame of audio through the pipeline (AGC →
/// [`super::handle_wake_word_detection`]), matching the per-frame processing in
/// [`run_streaming_detection`].
///
/// Partial chunks are passed through **without zero-padding** (mahbot-1006 G):
/// production feeds raw mic chunks as-is, and both the NS stage (which buffers
/// incomplete 160-sample frames internally) and `handle_wake_word_detection`
/// (which accumulates partial tails in `ctx.audio_buffer` until the next real
/// chunk completes a [`FRAME_LENGTH`](super::FRAME_LENGTH) frame) carry the
/// tail forward.  Zero-padding instead made the AGC adaptation cadence at
/// utterance boundaries differ from production.
///
/// Empty slices are a no-op (the preprocessor buffers nothing new and the
/// detection frame loop needs a complete frame) — silence must be fed as
/// actual [`FRAME_LENGTH`](super::FRAME_LENGTH) digital-silence chunks, which
/// is what production's mic delivers.
///
/// Extracted as a shared helper to eliminate the structural near-duplicate
/// between `feed_audio` and `run_streaming_detection` (mahbot-922, reviewer
/// feedback).
fn process_frame(samples: &[f32], ctx: &mut super::PipelineCtx) {
    let processed = ctx.audio_preprocessor.process(samples.to_vec());
    super::handle_wake_word_detection(&processed, ctx);
}

/// Feed audio through the production pipeline (AGC → [`super::handle_wake_word_detection`])
/// in [`FRAME_LENGTH`](super::FRAME_LENGTH) chunks, then send silence frames to flush any
/// remaining voice batch — matching the processing pattern in [`run_streaming_detection`].
///
/// This is used to pre-warm the pipeline (verifier, AGC) before the actual test utterance,
/// without contaminating latency measurements (mahbot-922).
fn feed_audio(samples: &[f32], ctx: &mut super::PipelineCtx) {
    for chunk in samples.chunks(super::FRAME_LENGTH) {
        process_frame(chunk, ctx);
    }
    // Feed digital-silence frames to flush any remaining voice batch (matches
    // the post-audio flush in run_streaming_detection).  Real zeros in
    // FRAME_LENGTH chunks, NOT empty Vecs: production's mic keeps delivering
    // 512-sample chunks of silence after speech, and the NS stage buffers
    // incomplete frames internally — an empty chunk would never produce a
    // complete frame for the detection loop (mahbot-1006 G).
    for _ in 0..3 {
        process_frame(&vec![0.0; super::FRAME_LENGTH], ctx);
    }
}

/// Consume the verifier warm-up period by feeding [`generate_warmup_noise`] through
/// [`feed_audio`].  After this call the verifier is active and the latency timer in
/// [`run_streaming_detection`] reflects only the test utterance (mahbot-922, mahbot-926).
///
/// ## Stride-8 warm-up (mahbot-926)
///
/// The stride-8 sliding-window embedding extraction (mahbot-923) requires at least
/// [`EMBEDDING_WINDOW_FRAMES`](super::EMBEDDING_WINDOW_FRAMES) mel frames (76) to
/// produce even one embedding.  With [`WARMUP_PREPEND_SAMPLES`] = 20480, the warm-up
/// noise produces ~125 mel frames → 7 stride-8 embeddings, safely exceeding the
/// verifier's [`VERIFIER_WARMUP_EMBEDDINGS`] = 4 requirement.
///
/// ## Diagnostics
///
/// - If VAD didn't trigger (fewer than [`VERIFIER_WARMUP_EMBEDDINGS`] embeddings
///   produced), a `warn!()` is emitted so maintainers know the warm-up wasn't fully
///   consumed.  Detection on short utterances (<1 s) will be disadvantaged.
/// - If the warm-up noise triggered a false detection, a `warn!()` is emitted and
///   the detection state is restored to prevent cooldown from corrupting subsequent
///   benchmark measurements (mahbot-922).
///
/// ## Residue clearing (mahbot-1003, mahbot-1006)
///
/// After feeding warm-up audio, this function clears warm-up residues that would
/// otherwise contaminate the subsequent test utterance's detection pipeline:
///
/// - **`mel_frame_buffer`** — cleared so the first stride-8 windows contain only
///   test utterance mel frames, not warm-up remnants + test audio mixed together.
/// - **`next_window_start`** — reset to 0 so embedding extraction starts from the
///   first test-utterance mel frame.
/// - **`score_window`** — cleared so warm-up classifier scores cannot create a
///   premature candidate on the first test embedding.
/// - **`candidate`** — cleared to discard any bounded classifier candidate that
///   may have formed after the warm-up embeddings exceeded
///   [`VERIFIER_WARMUP_EMBEDDINGS`].
/// - **`segment_silence_hops`** — reset to 0 so the test utterance starts a fresh
///   segment (the warm-up flush's VAD-negative hops must not shorten the test
///   utterance's trailing-silence window, mahbot-1006 H).
/// - **`audio_preprocessor`** — `reset()` (mahbot-1006 A): the warm-up drives the
///   AGC to a speech-adapted gain and the NS to a speech-adapted profile; both
///   must be fresh for the test utterance, matching training's lazy-init AGC and
///   production's `reset_detection_segment()` at segment boundaries.
///
/// **Preserved**: `embedding_ring` (keeps warm-up embeddings so
/// [`is_warmup_period`](crate::audio::voice_verifier::VoiceVerifier) returns
/// `false` and detections are not suppressed), `adaptive_threshold` (preserves
/// warm-up-adapted state — see mahbot-1006 F; the cold-start pass instead uses
/// a fresh [`AdaptiveThresholdState::new`]).
///
/// ## Warm-up ring-content divergence (mahbot-1006 E)
///
/// The preserved warm-up embeddings in `embedding_ring` are TTS speech
/// embeddings, whereas production's ring at a segment start contains ambient
/// noise/silence embeddings.  This is an accepted divergence for the warm pass:
/// the warm-up must produce embeddings at all (only VAD-triggering speech does
/// so reliably), and the classifier's first test windows are pure test content
/// because `mel_frame_buffer` is cleared.  The cold-start pass (mahbot-1006 D)
/// measures the empty-ring case instead, which is production's exact post-
/// silence state.
fn consume_warmup(ctx: &mut super::PipelineCtx) {
    let before_embeddings = ctx.embedding_ring.len();
    let before_detection = ctx.last_wake_word_detection;
    let noise = generate_warmup_noise();

    // ── Feed warm-up audio (single pass) ───────────────────────────────────
    // With TTS-generated speech, VAD triggers on almost every frame, producing
    // enough mel frames for ≥7 stride-8 embeddings.  A single pass suffices —
    // re-feeding identical audio through an already-adapted AGC cannot increase
    // VAD triggering (mahbot-926, reviewer feedback).
    feed_audio(&noise, ctx);
    let produced = ctx.embedding_ring.len().saturating_sub(before_embeddings);

    // Diagnostic: validate warm-up produced enough embeddings (mahbot-922/926).
    // When warm-up fails (TTS not available, pink noise used), this warning
    // indicates the verifier will NOT be active — every variant's first 4
    // embeddings will be consumed by warm-up suppression instead of scoring,
    // making detection impossible for short utterances (<1 s).
    if produced < crate::audio::voice_verifier::VERIFIER_WARMUP_EMBEDDINGS {
        warn!(
            "Warm-up produced only {} embedding(s) (need >= {}) — \
             verifier will NOT be active for this variant.  \
             This is expected when using the pink-noise fallback (TTS not available). \
             Every variant's first 4 embeddings will be consumed by warm-up suppression. \
             Detection on short utterances (<1 s) will be disadvantaged.",
            produced,
            crate::audio::voice_verifier::VERIFIER_WARMUP_EMBEDDINGS,
        );
    }

    // Guard: warm-up noise should not trigger a false detection.  If it does,
    // restore the pre-warm-up state so cooldown doesn't suppress all subsequent
    // benchmark measurements (reviewer feedback, mahbot-922).  We must also
    // reset `is_recording` because detection sets it to `true`, which would
    // suppress stride-8 scoring for the entire benchmark variant.
    if ctx.last_wake_word_detection != before_detection {
        warn!(
            "Warm-up triggered a false detection — restoring detection \
             state to prevent cooldown corruption.",
        );
        ctx.last_wake_word_detection = before_detection;
        ctx.is_recording = false;
    }

    // ── Clear warm-up residues before test utterance processing (mahbot-1003) ──
    // Warm-up audio produces mel frames, classifier scores, and may create a
    // bounded classifier candidate.  These residues must be cleared so the
    // test utterance starts with a clean detection slate.  Specifically:
    //
    // 1. Mel frame contamination — the first stride-8 windows would span
    //    warm-up remnants + test audio, producing mixed-content embeddings
    //    that score poorly and fall below NO_MATCH_RESET_THRESHOLD, clearing
    //    the score window before clean test embeddings arrive.
    //
    // 2. Premature candidate — preserved warm-up classifier scores may create
    //    a candidate on the first test embedding, consuming test frames without
    //    verifier confirmation, then expiring before enough clean frames
    //    remain to re-trigger.
    //
    // 3. Candidate from warm-up — a candidate may have formed after the
    //    warm-up embeddings exceeded VERIFIER_WARMUP_EMBEDDINGS.  Without
    //    clearing it, the candidate would consume test utterance frames
    //    during its update lifecycle.
    //
    // embedding_ring is preserved so is_warmup_period returns false for test
    // utterances — the verifier has context and detections are NOT suppressed.
    ctx.mel_frame_buffer.clear();
    ctx.next_window_start = 0;
    ctx.score_window.clear();
    ctx.candidate = None;
    ctx.segment_silence_hops = 0;

    // ── Reset instrumentation (mahbot-1005 §1) ─────────────────────────────
    // Warm-up audio passes through the full scoring pipeline and records into
    // ctx.instrumentation: per_frame_scores, peak_score, peak_verifier_score,
    // candidate lifecycle counters, VAD counts, and the adaptive threshold
    // trajectory.  Without a reset, warm-up-only scores contaminate the test
    // utterance's per-variant metrics:
    //
    //   - Silence/noise negatives report "classifier triggers" from warm-up
    //     speech (the fake "Silence: 1/1 trigger").
    //   - Inflated trigger counts across all categories.
    //   - Miss-classification buckets polluted by warm-up frames (a genuine
    //     classifier miss rebucketed as "verifier miss").
    //   - peak_verifier_score non-zero even for silence variants.
    //
    // The warm-up "head start" (mahbot-1002) remains observable via the two
    // warmup_* fields captured below — the values are preserved, not hidden.
    //
    // ctx.peak_score is also reset: although the streaming path never writes it
    // today, reset_detection_segment() flushes it into instrumentation, so any
    // future writer would re-contaminate the fresh instrumentation at the next
    // segment boundary (analyst feedback, mahbot-1005).
    let warmup_max_rolling_sum = max_rolling_sum(&ctx.instrumentation.per_frame_scores);
    let warmup_peak_verifier_score = ctx.instrumentation.peak_verifier_score;
    ctx.instrumentation = super::DetectionInstrumentation {
        warmup_max_rolling_sum,
        warmup_peak_verifier_score,
        ..super::DetectionInstrumentation::new()
    };
    ctx.peak_score = 0.0;

    // ── Fresh AGC/NS state for the test utterance (mahbot-1006 A) ─────────
    // The warm-up audio drives the AGC's asymmetric EMA to a speech-adapted
    // gain (~1× for ~1.28 s of TTS speech).  Production starts each detection
    // segment from a fresh AGC (reset_detection_segment calls
    // audio_preprocessor.reset()), and training uses fresh per-variant AGC, so
    // the benchmark must NOT let the warm-up-adapted gain carry into the test
    // utterance — a volume-down variant would otherwise be under-amplified for
    // the first few seconds (AGC release α=0.02, ~3.6 s to converge), biasing
    // exactly the volume-robustness variants this benchmark probes.
    //
    // reset() (not clear_buffer()): the critique on mahbot-1006 noted the two
    // differ in NS-profile preservation, and production segment boundaries use
    // reset() — the warm-up TTS speech should not seed the noise suppressor's
    // adapted profile for the test utterance either.
    //
    // Note (warm-up source reporting): when TTS warm-up is unavailable and the
    // pink-noise fallback fires, the fallback drives the AGC toward MIN_GAIN
    // (0.25×).  Without this reset that would guarantee VAD misses on short
    // utterances; with it the test utterance starts from lazy init like
    // training and production's digital-silence case.  The warm-up source is
    // reported in the JSON reproducibility block.
    ctx.audio_preprocessor.reset();
}

// ── Prerequisite check ─────────────────────────────────────────────────────

/// Ensure voice ONNX models are loaded.  Returns an error if the model
/// directory doesn't exist or loading fails, with a helpful message.
fn ensure_voice_models_loaded() -> Result<(), String> {
    if super::models_ready() {
        return Ok(());
    }

    let dir = super::model_dir().ok_or_else(|| {
        "Cannot resolve voice model directory. Is CONFIG.storage_root set?".to_string()
    })?;

    if !dir.join(super::MEL_MODEL_FILENAME).exists() {
        return Err(format!(
            "Mel spectrogram model not found at {}. Run the app to download models.",
            dir.join(super::MEL_MODEL_FILENAME).display()
        ));
    }
    if !dir.join(super::EMBED_MODEL_FILENAME).exists() {
        return Err(format!(
            "Embedding model not found at {}. Run the app to download models.",
            dir.join(super::EMBED_MODEL_FILENAME).display()
        ));
    }

    let models =
        super::load_onnx_models(&dir).map_err(|e| format!("Failed to load ONNX models: {e}"))?;
    super::ONNX_MODELS.set(models).map_err(|_| {
        "ONNX_MODELS already set by another test — cannot re-initialize".to_string()
    })?;
    super::MODELS_STATE.store(
        super::ModelState::Ready,
        std::sync::atomic::Ordering::Release,
    );

    info!("Voice models loaded from cache");
    Ok(())
}

/// Apply PCM transforms to enrollment variants (mahbot-932 Fix 5).
///
/// For each raw TTS enrollment variant, produces up to 5 sequences:
/// original, speed-down (0.95×), speed-up (1.05×, conditional on ≥500ms),
/// volume-down (-3dB), and pink noise (25dB SNR).  Matches the production
/// pipeline in [`prewarm_phrase_embeddings`](super::prewarm_phrase_embeddings).
fn pcm_augment_enrollment_variants(variants: &[(Vec<f32>, String)]) -> Vec<(Vec<f32>, String)> {
    let mut all = Vec::new();
    for (i, (pcm, label)) in variants.iter().enumerate() {
        // 1. Original (AGC'd, unmodified)
        all.push((pcm.clone(), format!("{label}_original")));

        // 2. Speed-down (0.95×)
        let speed_down =
            crate::audio::tts_data_gen::speed_perturbation(pcm, TARGET_SAMPLE_RATE, 0.95);
        all.push((speed_down, format!("{label}_speed_down")));

        // 3. Speed-up (1.05×, conditional — skip if too short)
        let pre_pad_samples = pcm.len().saturating_sub(2 * super::CONTEXT_PADDING_SAMPLES);
        let pre_pad_ms = (pre_pad_samples as u64 * 1000) / u64::from(TARGET_SAMPLE_RATE);
        if pre_pad_ms >= 500 {
            let speed_up =
                crate::audio::tts_data_gen::speed_perturbation(pcm, TARGET_SAMPLE_RATE, 1.05);
            all.push((speed_up, format!("{label}_speed_up")));
        }

        // 4. Volume-down (-3dB)
        let vol_down = crate::util::apply_gain(pcm, -3.0);
        all.push((vol_down, format!("{label}_vol_down")));

        // 5. Pink noise (SNR 25dB)
        let noise = crate::util::add_noise(pcm, 25.0, i as u64);
        all.push((noise, format!("{label}_noise")));
    }
    all
}

/// Generate owner-negative sequences (non-wake-word TTS phrases, mahbot-932 Fix 6).
///
/// These are TTS-synthesised phrases tagged as `Source::Owner` for use in
/// classifier and verifier training.  Documented limitation: TTS speech
/// cannot match the distribution of real human Phase 3 speech.
///
/// ## Preprocessing alignment (mahbot-1006 C/L, mahbot-1009)
///
/// Production captures owner negatives as real post-AGC/post-NS mic audio
/// (Phase 3 collection).  The benchmark's TTS surrogates are therefore routed
/// through the same pipeline as production's `prewarm_phrase_embeddings`
/// TTS-negative path so the classifier/verifier do not train on a raw-TTS
/// distribution that production never produces:
///
/// 1. **Fresh [`AudioPreprocessor`] per phrase × seed** (mahbot-1009) — fed in
///    [`FRAME_LENGTH`](super::FRAME_LENGTH) chunks as-is (no zero-padding),
///    matching the per-segment fresh-AGC distribution of the streaming path.
/// 2. **VAD gating** through a dedicated earshot detector per clip (never the
///    global `VAD_DETECTOR`) via [`super::vad_gate_streaming_mel`], so only
///    VAD-positive hops produce mel frames and windows anchor at speech onset.
/// 3. Dense stride-8 embeddings from the streaming-layout mel frames.
///
/// Divergence from production's `prewarm_phrase_embeddings`: this emits only
/// **variant 0** — production additionally derives 4 augmented PCM variants
/// (speed-down, speed-up, volume-down, noise) from the speech-only audio and
/// embeds each.  The benchmark deliberately omits them: owner negatives stand
/// in for real Phase 3 mic captures (raw, not TTS-augmented), so the single
/// VAD-gated original keeps the surrogate representative without inventing
/// augmented data production never collects.  The shared variant-0 path is
/// structurally identical (same preprocessor, same `vad_gate_streaming_mel`
/// wrapper over `process_streaming_frames_inner`, same embeddings helper).
///
/// The config comes from CONFIG (mahbot-1006 L) — identical to
/// `PreprocessorConfig::default()` under default settings, differs only when
/// a deployment disables NS/AGC.
fn generate_owner_negative_sequences(
    available_styles: &[String],
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Vec<EmbeddingSequence> {
    use crate::audio::audio_preprocessor::AudioPreprocessor;
    let num_styles = available_styles.len().max(1);
    let mut sequences = Vec::new();
    let config = enrollment_preprocessor_config();
    let chunk_size = super::FRAME_LENGTH;

    for (i, &phrase) in OWNER_NEGATIVE_PHRASES.iter().enumerate() {
        for seed in 0..3 {
            let style = &available_styles[(i * 3 + seed) % num_styles];
            let seed_val = 1000 + i as u64 * 3 + seed as u64;
            if let Some(pcm) = super::synthesize_with_pcm_cache(
                phrase,
                style,
                seed_val,
                TARGET_SAMPLE_RATE,
                model_hash,
                cache_dir,
            ) {
                // AGC/NS the TTS PCM through a fresh per-clip preprocessor,
                // fed in FRAME_LENGTH chunks as-is (matches
                // prewarm_phrase_embeddings, mahbot-1009).
                let mut pre = AudioPreprocessor::new(config);
                let mut agc_audio: Vec<f32> = Vec::with_capacity(pcm.len());
                for chunk in pcm.chunks(chunk_size) {
                    agc_audio.extend(pre.process(chunk.to_vec()));
                }
                // VAD-gate with a dedicated detector (never the global
                // VAD_DETECTOR) — only VAD-positive hops produce mel frames,
                // windows anchor at speech onset (matches the streaming path,
                // mahbot-1009).
                let mut detector = Detector::default();
                let (mel_frames, _speech_audio) =
                    super::vad_gate_streaming_mel(&agc_audio, |hop| {
                        super::is_speech_with_detector(hop, &mut detector, super::VAD_THRESHOLD)
                    });
                if mel_frames.is_empty() {
                    // Same guard as production prewarm_phrase_embeddings: no
                    // VAD-positive speech ⇒ no mel frames ⇒ skip the clip
                    // rather than train on a degenerate constant-silence
                    // embedding (pad_mel_frames_to_window's empty fallback).
                    warn!(
                        "Owner-negative '{phrase}' seed {seed}: no VAD-positive \
                         speech — skipping (matches production prewarm)"
                    );
                    continue;
                }
                let embs = super::embeddings_from_mel_frames(
                    super::ONNX_MODELS
                        .get()
                        .expect("ONNX models must be loaded by the benchmark"),
                    &mel_frames,
                );
                match embs {
                    Ok(embs) if !embs.is_empty() => {
                        sequences.push(EmbeddingSequence::negative(
                            UtteranceId {
                                sequence_index: i * 3 + seed,
                                variant_index: 0,
                            },
                            Source::Owner,
                            None,
                            embs,
                        ));
                    }
                    _ => warn!("Owner-negative '{phrase}' seed {seed}: no embeddings"),
                }
            }
        }
    }
    sequences
}

/// Generate ambient noise sequences for negative training (mahbot-932 Fix 7).
///
/// Produces `EmbeddingSequence` values tagged as [`Source::Ambient`] from noise
/// profiles at 2 SNR levels each.
///
/// ## Preprocessing alignment (mahbot-1006 C/L)
///
/// Production captures real ambient negatives as post-AGC/post-NS mic audio
/// collected during Phase 3.  The benchmark's synthetic noise profiles are
/// therefore routed through a **shared** [`AudioPreprocessor`] (matching the
/// persistent live-mic preprocessor) before embedding extraction, so the
/// classifier/verifier do not train on a raw-noise distribution production
/// never produces.  Documented limitation: synthetic noise through a fresh
/// noise suppressor is more aggressively attenuated than production's
/// room-adapted NS; this is a surrogate for real ambient capture.
fn generate_ambient_noise_sequences() -> Vec<EmbeddingSequence> {
    use crate::audio::audio_preprocessor::AudioPreprocessor;
    let mut sequences = Vec::new();
    let mut pre = AudioPreprocessor::new(enrollment_preprocessor_config());
    let chunk_size = super::FRAME_LENGTH;

    for (seq_idx, (label, noise_fn)) in NOISE_PROFILES.iter().enumerate() {
        let raw = noise_fn();
        // AGC/NS the raw noise (shared preprocessor, zero-padded chunks).
        let mut process_agc = |pcm: &[f32]| -> Vec<f32> {
            let mut agc_audio: Vec<f32> = Vec::with_capacity(pcm.len());
            for chunk in pcm.chunks(chunk_size) {
                let mut padded = chunk.to_vec();
                padded.resize(chunk_size, 0.0);
                agc_audio.extend(pre.process(padded));
            }
            agc_audio
        };
        // Level 1: full amplitude
        let agc_raw = process_agc(&raw);
        match super::process_enrollment_sample(&agc_raw) {
            Ok(embs) if !embs.is_empty() => {
                sequences.push(EmbeddingSequence::negative(
                    UtteranceId {
                        sequence_index: seq_idx * 2,
                        variant_index: 0,
                    },
                    Source::Ambient,
                    None,
                    embs,
                ));
            }
            _ => warn!("Ambient '{label}' level-0: no embeddings"),
        }
        // Level 2: reduced amplitude (-6dB)
        let attenuated = crate::util::apply_gain(&raw, -6.0);
        let agc_attenuated = process_agc(&attenuated);
        match super::process_enrollment_sample(&agc_attenuated) {
            Ok(embs) if !embs.is_empty() => {
                sequences.push(EmbeddingSequence::negative(
                    UtteranceId {
                        sequence_index: seq_idx * 2 + 1,
                        variant_index: 0,
                    },
                    Source::Ambient,
                    None,
                    embs,
                ));
            }
            _ => warn!("Ambient '{label}' level-1: no embeddings"),
        }
    }
    sequences
}

// ── Enrollment ────────────────────────────────────────────────────────────

/// Process a list of audio clips through the enrollment embedding pipeline.
///
/// Returns:
/// * `failed_count` — how many variants failed embedding extraction.
///   Process audio variants through a given embedding extraction function.
///
/// Shared helper that eliminates the duplicated per-variant loop
/// (mahbot-855 review).  Callers supply the extraction function — typically
/// [`super::process_enrollment_sample`] for dense stride-8 embeddings.
/// The old streaming path was removed in mahbot-923; the benchmark now uses
/// dense-only embeddings throughout.
///
/// Each variant becomes one [`EmbeddingSequence`] with the given `source`.
/// Compute VAD frame decisions and segment audio into utterances at the
/// enrollment VAD threshold.  Shared by the VAD-gated enrollment pipeline
/// and the VAD segmentation validation test to eliminate duplication.
///
/// # Panics
/// If called with zero-length audio or empty VAD decisions.
fn compute_vad_segments(audio: &[f32]) -> (Vec<bool>, Vec<Vec<f32>>) {
    let n_frames = audio.len().saturating_sub(super::FRAME_LENGTH) / super::HOP_LENGTH + 1;
    let mut detector = Detector::default();
    let mut vad_decisions: Vec<bool> = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let start = i * super::HOP_LENGTH;
        let end = (start + super::FRAME_LENGTH).min(audio.len());
        let frame = &audio[start..end];
        // Feed only the NEW HOP_LENGTH samples to avoid double-feeding
        // overlapping audio to earshot.  Each frame (at HOP_LENGTH stride)
        // overlaps the previous by 50% — feeding the full 512-sample frame
        // would duplicate 256 samples, corrupting earshot's ring buffer.
        // This must match the production VAD call pattern in both
        // process_streaming_frames_inner and handle_enrollment_audio
        // to maintain train-inference consistency (mahbot-900).
        vad_decisions.push(super::is_speech_with_detector(
            &frame[..super::HOP_LENGTH],
            &mut detector,
            super::VAD_THRESHOLD,
        ));
    }
    let utterances = super::segment_utterances_by_vad(
        audio,
        &vad_decisions,
        &super::DEFAULT_VAD_SEGMENTATION_CONFIG,
    );
    (vad_decisions, utterances)
}

/// Build the audio preprocessor configuration from the same CONFIG flags that
/// production uses (mahbot-1006 L).
///
/// `PipelineCtx::new()` constructs the live-mic preprocessor from
/// `CONFIG.voice_noise_suppression()` / `CONFIG.voice_agc()` (both default to
/// enabled when unset).  The benchmark's enrollment/training path previously
/// hardcoded `PreprocessorConfig::default()` (NS+AGC always on), which silently
/// diverged whenever a deployment disables either stage.
///
/// Delegates to [`super::preprocessor_config_from_config()`] — the same helper
/// `PipelineCtx::new()` uses — so the parsing semantics cannot drift between
/// the benchmark and production.
fn enrollment_preprocessor_config() -> crate::audio::audio_preprocessor::PreprocessorConfig {
    super::preprocessor_config_from_config()
}

/// Process enrollment clips through VAD-gated utterance segmentation with the
/// **production ordering** (mahbot-1006 B/I/L).
///
/// Production's enrollment path (live mic loop → `handle_enrollment_audio` →
/// `handle_enrollment_sample`) is:
///
/// ```text
/// raw mic audio → AGC/NS → VAD segment → augment after segmentation → embeddings
/// ```
///
/// The old benchmark path applied augmentation to the raw TTS PCM **before**
/// AGC (`raw → augment → AGC → VAD`), so:
/// - AGC partially normalized away the volume-down variant (re-amplifying it
///   back toward target) and NS partially removed the noise variant's added
///   noise, making those variants less challenging than production's
///   equivalents (mahbot-1006 B);
/// - VAD segmentation ran on the concatenated *augmented* variants, so a
///   variant could be silently dropped when VAD didn't fire on it, whereas
///   production runs VAD only on the original mic utterance and always includes
///   the augmented variants (mahbot-1006 I).
///
/// This function instead:
/// 1. Applies AGC/NS to each raw TTS clip through a **fresh**
///    [`AudioPreprocessor`] built from the CONFIG-driven
///    [`enrollment_preprocessor_config`] — per-clip fresh state matches
///    production's `reset_detection_segment()` fresh-AGC distribution
///    (mahbot-886 rationale, kept).
/// 2. Concatenates the AGC'd **originals** with 2.0 s silence gaps and
///    VAD-segments them (only originals, matching `handle_enrollment_audio`).
/// 3. Derives the 4 augmented variants (speed-down, speed-up conditional,
///    volume-down, noise) from each **AGC'd** utterance audio — exactly the
///    `handle_enrollment_sample` variant set (mahbot-878) — then extracts
///    embeddings from the original and all 4 variants.
///
/// Returns dense-only EmbeddingSequences (stride-8) for classifier and
/// verifier training.  The old streaming path was removed in mahbot-923.
#[allow(clippy::too_many_lines)]
fn vad_segment_and_enroll(enrollment_variants: &[(Vec<f32>, String)]) -> Vec<EmbeddingSequence> {
    use crate::audio::audio_preprocessor::AudioPreprocessor;
    let chunk_size = super::FRAME_LENGTH;
    let config = enrollment_preprocessor_config();

    // ── Per-clip AGC (mahbot-886, mahbot-1006 B) ──
    // Each clip processed through a fresh AudioPreprocessor (both AGC and
    // noise suppressor), matching the production detection path.  Shared-AGC
    // approach (concatenating variants then applying AGC) created a different
    // distribution — running_rms converged during training but detection starts
    // fresh, causing 46% miss rate on TTS variants.  Per-clip fresh NS avoids
    // the same training-inference mismatch.  Chunks are fed as-is (no
    // zero-padding): the NS buffers incomplete frames internally and the next
    // chunk (or the silence gap) completes them, matching production's tail
    // accumulation (mahbot-1006 G).
    let mut per_clip_agc: Vec<Vec<f32>> = Vec::with_capacity(enrollment_variants.len());
    for (samples, _label) in enrollment_variants {
        let mut pre = AudioPreprocessor::new(config);
        let mut processed: Vec<f32> = Vec::with_capacity(samples.len());
        for chunk in samples.chunks(chunk_size) {
            processed.extend(pre.process(chunk.to_vec()));
        }
        per_clip_agc.push(processed);
    }

    // ── Concatenate AGC'd originals with 2.0s silence gaps ──
    // 2.0s well exceeds ENROLLMENT_SILENCE_THRESHOLD_SAMPLES (~304ms after
    // mahbot-1001 Fix 7) for clean boundaries.
    let silence_gap_samples = (2.0 * f64::from(super::SAMPLE_RATE)) as usize;
    let silence: Vec<f32> = vec![0.0f32; silence_gap_samples];

    let mut combined_audio: Vec<f32> = Vec::new();
    for processed in &per_clip_agc {
        if !combined_audio.is_empty() {
            combined_audio.extend_from_slice(&silence);
        }
        combined_audio.extend_from_slice(processed);
    }
    // Trailing silence for the last utterance
    combined_audio.extend_from_slice(&silence);

    info!(
        "VAD concatenation: {} total samples ({:.1}s) from {} originals (per-clip AGC, mahbot-1006 B)",
        combined_audio.len(),
        combined_audio.len() as f64 / f64::from(super::SAMPLE_RATE),
        enrollment_variants.len(),
    );

    // ── VAD segmentation on the ORIGINALS only (mahbot-1006 I) ──
    let (_vad_decisions, utterances) = compute_vad_segments(&combined_audio);
    info!(
        "VAD segmentation: {} utterances from {} concatenated originals",
        utterances.len(),
        enrollment_variants.len(),
    );

    // ── Augment AFTER segmentation, then extract embeddings (mahbot-1006 B) ──
    // Each VAD utterance is treated like production's `handle_enrollment_sample`
    // input: the 4 augmented PCM variants are derived from the AGC'd utterance
    // audio, and all 5 variants are embedded.  Every variant is always included
    // — none can be silently dropped by VAD (which only gated the original).
    let mut dense_sequences: Vec<EmbeddingSequence> = Vec::new();

    for (i, utterance) in utterances.iter().enumerate() {
        let speed_down =
            crate::audio::tts_data_gen::speed_perturbation(utterance, TARGET_SAMPLE_RATE, 0.95);
        // Speed-up is conditional on ≥500 ms unpadded duration (matches
        // handle_enrollment_sample's skip_speed_up).
        let pre_pad_duration_samples = utterance
            .len()
            .saturating_sub(2 * super::CONTEXT_PADDING_SAMPLES);
        let pre_pad_duration_ms =
            (pre_pad_duration_samples as u64 * 1000) / u64::from(TARGET_SAMPLE_RATE);
        let speed_up = if pre_pad_duration_ms >= 500 {
            Some(crate::audio::tts_data_gen::speed_perturbation(
                utterance,
                TARGET_SAMPLE_RATE,
                1.05,
            ))
        } else {
            None
        };
        let volume_down = crate::util::apply_gain(utterance, -3.0);
        let noise = crate::util::add_noise(utterance, 25.0, i as u64);

        info!(
            "Utterance {i}: {} samples ({:.2}s) → original + speed_down + {} + vol_down + noise",
            utterance.len(),
            utterance.len() as f64 / f64::from(super::SAMPLE_RATE),
            if speed_up.is_some() {
                "speed_up".to_string()
            } else {
                "speed_up(skipped,<500ms)".to_string()
            },
        );

        // Helper closure: push one variant's embeddings (mahbot-878 naming:
        // variant 0 = original, 1 = speed-down, 2 = speed-up, 3 = volume-down,
        // 4 = noise).
        let push_variant = |variant_index: usize,
                            source: Source,
                            aug_family: Option<AugmentationFamily>,
                            pcm: &[f32],
                            sequences: &mut Vec<EmbeddingSequence>| {
            match super::process_enrollment_sample(pcm) {
                Ok(embeddings) if !embeddings.is_empty() => {
                    sequences.push(EmbeddingSequence::positive(
                        UtteranceId {
                            sequence_index: i,
                            variant_index,
                        },
                        source,
                        aug_family,
                        embeddings,
                    ));
                }
                Ok(_) => warn!("Utterance {i} variant {variant_index}: no embeddings extracted"),
                Err(e) => {
                    warn!(
                        "Utterance {i} variant {variant_index}: embedding extraction failed: {e}"
                    );
                }
            }
        };

        push_variant(0, Source::Enrollment, None, utterance, &mut dense_sequences);
        push_variant(
            1,
            Source::Augmentation,
            Some(AugmentationFamily::SpeedDown),
            &speed_down,
            &mut dense_sequences,
        );
        if let Some(ref spd_up) = speed_up {
            push_variant(
                2,
                Source::Augmentation,
                Some(AugmentationFamily::SpeedUp),
                spd_up,
                &mut dense_sequences,
            );
        }
        push_variant(
            3,
            Source::Augmentation,
            Some(AugmentationFamily::Volume),
            &volume_down,
            &mut dense_sequences,
        );
        push_variant(
            4,
            Source::Augmentation,
            Some(AugmentationFamily::Noise),
            &noise,
            &mut dense_sequences,
        );
    }

    info!(
        "VAD-gated enrollment (mahbot-1006 B): {} dense embeddings from {} sequences ({} utterances × ~5 variants)",
        dense_sequences
            .iter()
            .map(|s| s.embeddings.len())
            .sum::<usize>(),
        dense_sequences.len(),
        utterances.len(),
    );

    dense_sequences
}

// ── Streaming detection ─────────────────────────────

/// Result from [`run_streaming_detection`].
struct DetectionResult {
    detected: bool,
    /// Latency in milliseconds from feed start to detection (only meaningful
    /// when `detected` is true).
    latency_ms: Option<f64>,
    /// AGC convergence state captured at the end of the test utterance (before
    /// the trailing silence flush).  `None` when insufficient AGC-active
    /// frames.  Captured pre-flush so a segment boundary fired during the
    /// trailing silence cannot erase the utterance's convergence evidence
    /// (mahbot-1006 A/H).
    agc_converged: Option<bool>,
    /// Adaptive threshold state captured at the end of the test utterance
    /// (before the trailing silence flush).  A segment boundary fired during
    /// the flush calls `reset_detection_segment`, which resets
    /// `ctx.adaptive_threshold` to bootstrap; carrying this pre-flush snapshot
    /// forward lets the benchmark's shared-state design (mahbot-845) preserve
    /// the adaptation accumulated from the utterance's frames instead of
    /// re-bootstrapping per variant (mahbot-1006 F).
    adaptive_state_pre_flush: super::AdaptiveThresholdState,
}

/// Run the production streaming wake word detection pipeline on audio samples.
///
/// Feeds audio through [`handle_wake_word_detection`] in FRAME_LENGTH chunks,
/// exercising the full streaming chain: VAD gating, batch accumulation,
/// [`flush_voice_batch`], [`score_stride8_window`],
/// [`score_single_embedding`], and cooldown logic.
///
/// After all audio is fed, a silence frame is sent to flush any remaining
/// voice batch (matching how the production pipeline handles speech→silence
/// transitions).
///
/// Returns a [`DetectionResult`] with a flag and optional latency measurement.
fn run_streaming_detection(samples: &[f32], ctx: &mut super::PipelineCtx) -> DetectionResult {
    let feed_start = Instant::now();
    // Save pre-existing timestamp — we only return true if detection fires
    // during THIS call, not because a prior call already set the field.
    let before = ctx.last_wake_word_detection;

    // Feed audio in FRAME_LENGTH chunks, processing each through the
    // audio preprocessor (AGC + noise suppression) to match the production
    // pipeline (mahbot-856).  Without AGC, quiet variants like
    // M3.json_aug4_volume (-6dB reduction) are too faint for VAD to trigger,
    // producing false 0.00 detection scores.
    for chunk in samples.chunks(super::FRAME_LENGTH) {
        process_frame(chunk, ctx);
        if ctx.last_wake_word_detection != before {
            let latency = feed_start.elapsed().as_secs_f64() * 1000.0;
            return DetectionResult {
                detected: true,
                latency_ms: Some(latency),
                agc_converged: ctx.audio_preprocessor.agc_converged(),
                // No flush ran on this path, so the pre-flush snapshot is the
                // current state; the boundary-fire branch in the callers is
                // unreachable when `detected` is true anyway.
                adaptive_state_pre_flush: ctx.adaptive_threshold.clone(),
            };
        }
    }

    // ── Trailing silence flush (mahbot-1006 H) ───────────────────────────
    // Production continues on natural silence after the utterance and
    // detection can still fire for up to SEGMENT_TIMEOUT_HOPS (19) VAD-negative
    // hops before the segment-boundary reset clears the ring and per-segment
    // state.  The old flush fed exactly 3×512 zero frames (~96 ms), which
    // missed wake words whose rolling sum crossed threshold during the longer
    // production window (~304 ms).
    //
    // The AGC convergence evidence is captured BEFORE the flush: a segment
    // boundary fired during the flush resets the preprocessor (mahbot-1006 A)
    // and its voice-tests instrumentation, which would otherwise erase the
    // utterance's convergence history from `agc_converged()`.  The adaptive
    // threshold state is captured for the same reason — the boundary resets it
    // to bootstrap, and the benchmark's shared-state design (mahbot-845) wants
    // the adapted state to survive the boundary rather than re-bootstrapping
    // per variant (mahbot-1006 F).
    let agc_converged = ctx.audio_preprocessor.agc_converged();
    let adaptive_state_pre_flush = ctx.adaptive_threshold.clone();
    //
    // Feed FRAME_LENGTH digital-silence chunks (production's mic delivers real
    // zeros, not empty Vecs — see process_frame) until:
    //   1. detection fires, or
    //   2. the segment boundary fires, or
    //   3. we exceed the production window (SEGMENT_TIMEOUT_HOPS hops plus a
    //      small allowance for any remaining VAD-positive utterance tail).
    //
    // The boundary is detected via the embedding ring: reset_detection_segment
    // clears it (and no other non-detection path does within a variant), so a
    // non-empty→empty transition means the segment ended and further silence
    // cannot produce a detection.
    let mut silence_chunks = 0usize;
    while silence_chunks < super::SEGMENT_TIMEOUT_HOPS + 4 {
        if ctx.last_wake_word_detection != before {
            let latency = feed_start.elapsed().as_secs_f64() * 1000.0;
            return DetectionResult {
                detected: true,
                latency_ms: Some(latency),
                agc_converged,
                adaptive_state_pre_flush,
            };
        }
        let ring_was_nonempty = !ctx.embedding_ring.is_empty();
        process_frame(&vec![0.0; super::FRAME_LENGTH], ctx);
        silence_chunks += 1;
        // Segment boundary fired: the ring was cleared — further silence
        // cannot produce a detection, so stop feeding.
        if ring_was_nonempty && ctx.embedding_ring.is_empty() {
            break;
        }
    }

    DetectionResult {
        detected: ctx.last_wake_word_detection != before,
        latency_ms: None,
        agc_converged,
        adaptive_state_pre_flush,
    }
}

// ── Metrics reporting ────────────────────────────────────────────────────

/// Result for a single wake word variant during detection.
#[derive(Debug, Clone)]
struct PerVariantResult {
    /// Variant label (e.g. "M1.json seed 100" or "augmented speed_0.9").
    variant: String,
    /// Whether the variant triggered wake word detection.
    detected: bool,
    /// Peak per-frame total_score (classifier sigmoid output, range [0,1])
    /// achieved during processing. This is NOT the rolling sum — it's the
    /// maximum single-frame score. For rolling sum analysis, use the
    /// `per_frame_scores` field or `max_rolling_sum`.
    peak_score: f32,
    /// Maximum rolling sum (sum of 3 consecutive total_scores with decay)
    /// achieved during processing. Derived from per_frame_scores.
    max_rolling_sum: f32,
    /// Peak verifier prediction score achieved during processing
    /// (0.0 if verifier is untrained or no embeddings passed the threshold).
    verifier_score: f32,
    /// Number of embeddings produced during streaming detection (mahbot-886).
    /// This is the length of the verifier's embedding ring buffer after
    /// processing, which directly reflects how many mel frames passed through
    /// the embedding model (mahbot-922).
    n_embeddings: usize,
    /// Number of frames where total_score < NO_MATCH_RESET_THRESHOLD (0.316).
    n_frames_below_reset: usize,
    /// Whether the AGC converged to a stable gain level by utterance end.
    /// `Some(true)` if converged, `Some(false)` if not, `None` if insufficient
    /// data (< 20 AGC-active frames).
    agc_converged: Option<bool>,
    /// Count of VAD-positive 512-sample frames during streaming detection.
    vad_speech_frames: usize,
    /// Per-frame `[total_score, rolling_sum, threshold]` triples from classifier
    /// scoring (mahbot-891).  Use `ROLLING_SUM_IDX` / `THRESHOLD_IDX` for named
    /// access to the rolling sum and effective threshold fields respectively.
    per_frame_scores: Vec<[f32; 3]>,
    /// Whether the verifier warm-up was completed before this variant's test
    /// utterance was processed (embedding_ring had ≥
    /// [`VERIFIER_WARMUP_EMBEDDINGS`](crate::audio::voice_verifier::VERIFIER_WARMUP_EMBEDDINGS)
    /// entries after `consume_warmup`).  When `false`, the verifier will NOT be
    /// active, and the first 4 embeddings from the test utterance are consumed
    /// by warm-up suppression (mahbot-947).
    warmup_completed: bool,
    /// Number of embeddings in the ring buffer after warm-up consumption,
    /// before processing the test utterance (mahbot-947).
    warmup_n_embeddings: usize,
    /// Per-frame peak verifier score, parallel to `per_frame_scores`
    /// (mahbot-1005 §4).  0.0 per-frame means the verifier was not evaluated on
    /// that frame (warm-up period / untrained verifier), not a zero-confidence
    /// rejection.
    verifier_score_trajectory: Vec<f32>,
    /// Number of classifier candidates created during this session (mahbot-1005 §4).
    candidates_created: usize,
    /// Number of candidates confirmed (detection fired via the verifier gate).
    candidates_confirmed: usize,
    /// Number of candidates cleared without confirmation (expired via
    /// CANDIDATE_MAX_EMBEDDINGS or reset by a below-reset score frame).
    candidates_expired: usize,
    /// Index of the first classifier-trigger frame within `per_frame_scores`.
    /// `None` when the classifier never triggered on test-utterance frames.
    first_trigger_frame_idx: Option<usize>,
    /// Max rolling sum achieved by warm-up audio only (mahbot-1005 §1).
    warmup_max_rolling_sum: f32,
    /// Peak verifier score achieved by warm-up audio only (mahbot-1005 §1).
    warmup_peak_verifier_score: f32,
    /// Number of frames where the classifier triggered but detection was
    /// suppressed by verifier warm-up during the test run.
    n_warmup_suppressed_frames: usize,
    /// Number of embeddings attributed to the test utterance — the length of
    /// `per_frame_scores` (one entry per scored embedding after the warm-up
    /// instrumentation reset).  Unlike `n_embeddings` (ring length at end,
    /// which includes the intentionally preserved warm-up embeddings and can be
    /// cleared by a segment boundary), this counts only test-utterance frames.
    n_test_embeddings: usize,
    /// Verifier decision threshold in effect during this variant's processing.
    /// Used as the `threshold_at_peak` evidence for miss classification.
    verifier_threshold: f32,
    /// Whether the verifier was trained (and therefore the second-stage gate
    /// was active) during this variant's processing.  When `false`, the
    /// verifier is a no-op and a miss after crossing the effective threshold
    /// is a candidate lifecycle/timing issue, not a verifier rejection.
    verifier_trained: bool,
    /// Per-frame adaptive threshold trajectory (parallel to `per_frame_scores`,
    /// mahbot-1005 §8).  Previously collected in `DetectionInstrumentation`
    /// but never emitted in the benchmark report.
    adaptive_threshold_trajectory: Vec<f32>,
    /// Count of frames where the effective threshold hit ADAPTIVE_CEILING
    /// (mahbot-1005 §8).
    ceiling_limited_frames: usize,
    /// Detection latency in ms for this variant (only meaningful when
    /// `detected` is true; `None` otherwise).  Emitted per-variant so latency
    /// outliers can be root-caused (mahbot-1005 §8).
    latency_ms: Option<f64>,
}

/// Build a [`PerVariantResult`] from a completed [`run_streaming_detection`]
/// session.  Shared by `test_detection_samples` and `run_noise_overlap_test`
/// so the instrumentation-derived fields are populated identically everywhere
/// (mahbot-1005 §9: negatives must carry the same detail as positives).
fn build_per_variant_result(
    label: &str,
    result: &DetectionResult,
    ctx: &super::PipelineCtx,
    warmup_completed: bool,
    warmup_n_embeddings: usize,
    verifier: &VoiceVerifier,
) -> PerVariantResult {
    let peak = ctx.instrumentation.peak_score;
    let max_rs = max_rolling_sum(&ctx.instrumentation.per_frame_scores);
    // Test-utterance embedding count.  `per_frame_scores` is reset at the end
    // of consume_warmup (mahbot-1005 §1) and grows one entry per scored
    // embedding, so its length IS the test-utterance embedding count.  The
    // alternative (ring length minus warm-up) is unreliable because a segment
    // boundary during/after the test clears the embedding ring.
    let n_test_embeddings = ctx.instrumentation.per_frame_scores.len();
    PerVariantResult {
        variant: label.to_string(),
        detected: result.detected,
        peak_score: peak,
        max_rolling_sum: max_rs,
        verifier_score: ctx.instrumentation.peak_verifier_score,
        n_embeddings: ctx.embedding_ring.len(),
        n_frames_below_reset: ctx.instrumentation.n_frames_below_reset,
        // Captured pre-flush in run_streaming_detection (mahbot-1006 A/H) —
        // reading ctx.audio_preprocessor.agc_converged() here could report
        // None for a miss whose segment boundary reset the preprocessor.
        agc_converged: result.agc_converged,
        vad_speech_frames: ctx.instrumentation.vad_speech_frames,
        per_frame_scores: ctx.instrumentation.per_frame_scores.clone(),
        warmup_completed,
        warmup_n_embeddings,
        verifier_score_trajectory: ctx.instrumentation.per_frame_verifier_scores.clone(),
        candidates_created: ctx.instrumentation.candidates_created,
        candidates_confirmed: ctx.instrumentation.candidates_confirmed,
        candidates_expired: ctx.instrumentation.candidates_expired,
        first_trigger_frame_idx: ctx.instrumentation.first_trigger_frame_idx,
        warmup_max_rolling_sum: ctx.instrumentation.warmup_max_rolling_sum,
        warmup_peak_verifier_score: ctx.instrumentation.warmup_peak_verifier_score,
        n_warmup_suppressed_frames: ctx.instrumentation.n_warmup_suppressed_frames,
        n_test_embeddings,
        verifier_threshold: verifier.threshold,
        verifier_trained: verifier.is_trained(),
        adaptive_threshold_trajectory: ctx.instrumentation.adaptive_threshold_trajectory.clone(),
        ceiling_limited_frames: ctx.instrumentation.ceiling_limited_frames,
        latency_ms: result.latency_ms,
    }
}

// ── Exhaustive per-variant miss verdicts (mahbot-1005 §2) ─────────────────

/// Exhaustive per-variant verdict for positive (wake word) test variants.
///
/// Replaces the old 3-way bucket (classifier / adaptive_threshold / verifier)
/// which collapsed at least four distinct failure modes into a single
/// "verifier" label and mis-bucketed zero-embedding misses (VAD never fired)
/// as "classifier".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MissVerdict {
    /// Wake word detected on this variant.
    Detected,
    /// VAD produced no speech frames (or the test utterance produced zero
    /// embeddings) — the classifier never saw test audio.
    VadFailure,
    /// Rolling sum never reached [`MIN_CLASSIFIER_THRESHOLD`] (2.13) on
    /// test-utterance frames only.
    ClassifierNoTrigger,
    /// Crossed the hard floor ([`MIN_CLASSIFIER_THRESHOLD`] = 2.13) but never
    /// crossed the effective adaptive threshold.
    AdaptiveThresholdBlocked,
    /// Crossed the effective threshold; a candidate existed but the verifier
    /// peak stayed below the verifier decision threshold.
    VerifierRejected,
    /// Crossed the effective threshold and the verifier peak reached the
    /// verifier threshold, yet detection did not fire — the candidate expired
    /// ([`CANDIDATE_MAX_EMBEDDINGS`] exceeded) or the audio ended before
    /// confirmation.  Also covers an untrained (no-op) verifier, where no
    /// second-stage gate exists to reject.
    VerifierTiming,
    /// Warm-up was incomplete when the classifier triggered; the trigger was
    /// suppressed by verifier warm-up ([`VERIFIER_WARMUP_EMBEDDINGS`]).
    WarmupSuppression,
    /// AGC explicitly reported non-convergence (`Some(false)`).
    AgcFailure,
}

impl MissVerdict {
    /// Stable snake_case label for JSON output.
    fn as_str(self) -> &'static str {
        match self {
            Self::Detected => "detected",
            Self::VadFailure => "vad_failure",
            Self::ClassifierNoTrigger => "classifier_no_trigger",
            Self::AdaptiveThresholdBlocked => "adaptive_threshold_blocked",
            Self::VerifierRejected => "verifier_rejected",
            Self::VerifierTiming => "verifier_timing",
            Self::WarmupSuppression => "warmup_suppression",
            Self::AgcFailure => "agc_failure",
        }
    }
}

/// Classify a positive variant into an exhaustive verdict (mahbot-1005 §2).
///
/// ## Decision procedure (precedence, highest first)
///
/// 1. `detected` — no miss.
/// 2. `vad_failure` — VAD never fired (zero speech frames) OR no embeddings
///    were scored during the test utterance (`per_frame_scores` is empty after
///    the warm-up instrumentation reset).  Score-based verdicts are meaningless
///    when the classifier never saw test audio.  This fixes the old
///    mis-bucketing where a zero-embedding miss was labeled "classifier".
/// 3. `warmup_suppression` — warm-up incomplete AND the classifier triggered.
///    The trigger was structurally suppressed before scoring, which is a more
///    fundamental explanation than any score/AGC issue.
/// 4. `agc_failure` — AGC explicitly reported `Some(false)`.  `None`
///    (insufficient AGC-active frames) is NOT `agc_failure` — short utterances
///    fall through to the score-based verdicts.
/// 5. `classifier_no_trigger` — never reached [`MIN_CLASSIFIER_THRESHOLD`]
///    (2.13) on test-utterance frames.
/// 6. `adaptive_threshold_blocked` — reached the hard floor (2.13) but never
///    reached the per-frame effective adaptive threshold.  "Hard floor" here is
///    [`MIN_CLASSIFIER_THRESHOLD`] (the classifier-trigger condition), NOT
///    [`ADAPTIVE_FLOOR`](super::ADAPTIVE_FLOOR) (the adaptive threshold's own
///    clamp floor, which is an internal detail of the adaptive state).
/// 7. `verifier_timing` / `verifier_rejected` — crossed the effective
///    threshold; split by whether the verifier peak reached the verifier
///    threshold (lifecycle/timing failure) or not (genuine rejection).
fn classify_miss(pv: &PerVariantResult) -> MissVerdict {
    if pv.detected {
        return MissVerdict::Detected;
    }
    if pv.vad_speech_frames == 0 || pv.per_frame_scores.is_empty() {
        return MissVerdict::VadFailure;
    }
    let triggered = pv.max_rolling_sum >= MIN_CLASSIFIER_THRESHOLD;
    if triggered && !pv.warmup_completed {
        return MissVerdict::WarmupSuppression;
    }
    if pv.agc_converged == Some(false) {
        return MissVerdict::AgcFailure;
    }
    if !triggered {
        return MissVerdict::ClassifierNoTrigger;
    }
    let crossed_effective = pv
        .per_frame_scores
        .iter()
        .any(|s| s[ROLLING_SUM_IDX] >= s[THRESHOLD_IDX]);
    if !crossed_effective {
        return MissVerdict::AdaptiveThresholdBlocked;
    }
    if !pv.verifier_trained {
        // No second-stage gate: a miss after crossing the effective threshold
        // is a candidate lifecycle/timing issue (expired or audio ended).
        return MissVerdict::VerifierTiming;
    }
    if pv.verifier_score >= pv.verifier_threshold {
        MissVerdict::VerifierTiming
    } else {
        MissVerdict::VerifierRejected
    }
}

/// Extract the maximum rolling sum from per-frame score triples.
///
/// The rolling sum (`ROLLING_SUM_IDX`) is the accumulated classifier score with
/// decay over consecutive speech frames — the value that the adaptive threshold
/// ultimately gates.
fn max_rolling_sum(scores: &[[f32; 3]]) -> f32 {
    scores
        .iter()
        .map(|s| s[ROLLING_SUM_IDX])
        .fold(0.0_f32, f32::max)
}

/// Track per-variant detection results for reporting.
#[derive(Debug, Default)]
struct DetectionMetrics {
    total: usize,
    detected: usize,
    false_accepts: Vec<String>,
    /// Latency samples from true-positive detections (ms).
    latencies: Vec<f64>,
    /// Per-variant detail for positive detection logging (mahbot-845).
    per_variant: Vec<PerVariantResult>,
}

impl DetectionMetrics {
    fn detection_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.detected as f64 / self.total as f64
        }
    }

    fn mean_latency(&self) -> Option<f64> {
        if self.latencies.is_empty() {
            None
        } else {
            Some(self.latencies.iter().copied().sum::<f64>() / self.latencies.len() as f64)
        }
    }

    fn median_latency(&self) -> Option<f64> {
        let mut sorted = self.latencies.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        if n == 0 {
            None
        } else if n.is_multiple_of(2) {
            Some(f64::midpoint(sorted[n / 2 - 1], sorted[n / 2]))
        } else {
            Some(sorted[n / 2])
        }
    }

    fn p95_latency(&self) -> Option<f64> {
        let mut sorted = self.latencies.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        if n == 0 {
            None
        } else {
            let idx = ((n as f64) * 0.95).ceil() as usize - 1;
            Some(sorted[idx.min(n - 1)])
        }
    }
}

// ── Classifier trigger metrics (mahbot-952) ─────────────────────────────────

/// Classifier-only trigger metrics for a set of negative variants.
///
/// These metrics track the classifier stage independently from the
/// verifier, allowing separate monitoring of classifier behavior and
/// verifier filtering effectiveness run over run.
///
/// The four counters decompose what happens when the classifier triggers:
///
/// | Condition | Counter |
/// |-----------|---------|
/// | `max_rolling_sum < MIN_CLASSIFIER_THRESHOLD` | not counted |
/// | Triggered + `warmup_completed == false` | `warmup_suppressed` |
/// | Triggered + warmup completed + `detected == false` | `verifier_caught` |
/// | Triggered + `detected == true` | `full_pipeline_fa` |
#[derive(Debug, Default, Clone, Copy)]
struct ClassifierTriggerMetrics {
    /// Total variants tested in this group.
    total_variants: usize,
    /// Number of variants where max_rolling_sum >= MIN_CLASSIFIER_THRESHOLD.
    classifier_triggers: usize,
    /// Number where classifier triggered but warmup was not completed,
    /// so the verifier never evaluated these frames.
    warmup_suppressed: usize,
    /// Number where classifier triggered, warmup completed, but
    /// detected == false (verifier successfully rejected).
    verifier_caught: usize,
    /// Number where detected == true (full pipeline false accept).
    full_pipeline_fa: usize,
}

impl ClassifierTriggerMetrics {
    /// Compute metrics from a slice of per-variant results.
    fn compute(per_variant: &[PerVariantResult]) -> Self {
        let mut m = Self::default();
        for pv in per_variant {
            m.accumulate(pv);
        }
        m
    }

    /// The verifier's prevention effectiveness: fraction of classifier triggers
    /// that reached a fully-active verifier AND were rejected by it (i.e. did
    /// NOT become full-pipeline false accepts).  Warm-up suppression is EXCLUDED
    /// (mahbot-1005 §7): it is a pipeline-timing artefact, not verifier
    /// discrimination, and mixing it in inflated the old metric.
    ///
    /// Returns `None` when there are no verifier-evaluated triggers
    /// (division by zero).
    fn prevention_rate(&self) -> Option<f64> {
        let prevented = self.verifier_caught;
        let evaluated = self.verifier_caught + self.full_pipeline_fa;
        if evaluated == 0 {
            None
        } else {
            Some(prevented as f64 / evaluated as f64)
        }
    }

    /// Fraction of classifier triggers suppressed because the verifier warm-up
    /// was incomplete (mahbot-1005 §7).  Reported separately from
    /// [`prevention_rate`](Self::prevention_rate) so warm-up suppression is
    /// observable without being attributed to the verifier.
    fn warmup_suppression_rate(&self) -> Option<f64> {
        if self.classifier_triggers == 0 {
            None
        } else {
            Some(self.warmup_suppressed as f64 / self.classifier_triggers as f64)
        }
    }

    /// Accumulate a single per-variant result into this metrics struct.
    ///
    /// This shares the classification branching logic with [`compute`] so that
    /// callers who cannot collect a flat slice (e.g. per-tier partitioning) can
    /// reuse the same counting rules without duplication.
    fn accumulate(&mut self, pv: &PerVariantResult) {
        self.total_variants += 1;
        if pv.max_rolling_sum >= MIN_CLASSIFIER_THRESHOLD {
            self.classifier_triggers += 1;
            if !pv.warmup_completed {
                self.warmup_suppressed += 1;
            } else if !pv.detected {
                self.verifier_caught += 1;
            } else {
                self.full_pipeline_fa += 1;
            }
        }
    }
}

/// Verifier recall on positive (wake-word) test variants (mahbot-1008 Fix 6).
///
/// The fraction of genuine wake-word variants the verifier ACCEPTED among
/// those it actually evaluated: verifier trained + warm-up complete + the
/// classifier crossed the effective adaptive threshold (so a candidate existed
/// and the verifier gate was the deciding factor).  A variant counts as
/// accepted when its peak verifier score reached the verifier decision
/// threshold.
///
/// Returns `None` when the verifier was untrained or no variant reached the
/// verifier gate (division by zero).  Otherwise returns `(accepted,
/// evaluated)`.
fn verifier_recall(per_variant: &[PerVariantResult]) -> Option<(usize, usize)> {
    let mut evaluated = 0usize;
    let mut accepted = 0usize;
    for pv in per_variant {
        if !pv.verifier_trained || !pv.warmup_completed {
            continue;
        }
        let crossed_effective = pv
            .per_frame_scores
            .iter()
            .any(|s| s[ROLLING_SUM_IDX] >= s[THRESHOLD_IDX]);
        if !crossed_effective {
            continue;
        }
        evaluated += 1;
        if pv.verifier_score >= pv.verifier_threshold {
            accepted += 1;
        }
    }
    if evaluated == 0 {
        None
    } else {
        Some((accepted, evaluated))
    }
}

/// Convert classifier trigger metrics to a JSON object for the benchmark report.
fn ct_to_json(ct: &ClassifierTriggerMetrics) -> serde_json::Value {
    serde_json::json!({
        "total_variants": ct.total_variants,
        "classifier_triggers": ct.classifier_triggers,
        "warmup_suppressed": ct.warmup_suppressed,
        "verifier_caught": ct.verifier_caught,
        "full_pipeline_fa": ct.full_pipeline_fa,
        "prevention_rate": ct.prevention_rate(),
        "warmup_suppression_rate": ct.warmup_suppression_rate(),
    })
}

/// Full per-variant diagnostics for BOTH positive and negative variants
/// (mahbot-1005 §9).  Negatives historically omitted per-frame scores, verifier
/// trajectories, VAD/AGC detail and trigger-frame info — false-accept
/// root-causing was impossible without them.  For positives (`category: None`)
/// the exhaustive miss verdict and rejection-margin evidence are added.
// Clippy: this is a pure JSON serialization shim — one insert per field; the
// line count is inherent to the schema, not control flow worth splitting.
#[expect(clippy::too_many_lines)]
fn pv_to_json(pv: &PerVariantResult, category: Option<&str>) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("variant".to_string(), serde_json::json!(pv.variant));
    if let Some(cat) = category {
        obj.insert("category".to_string(), serde_json::json!(cat));
    }
    obj.insert("detected".to_string(), serde_json::json!(pv.detected));
    obj.insert("peak_score".to_string(), serde_json::json!(pv.peak_score));
    obj.insert(
        "max_rolling_sum".to_string(),
        serde_json::json!(pv.max_rolling_sum),
    );
    obj.insert(
        "verifier_score".to_string(),
        serde_json::json!(pv.verifier_score),
    );
    obj.insert(
        "n_embeddings".to_string(),
        serde_json::json!(pv.n_embeddings),
    );
    obj.insert(
        "n_test_embeddings".to_string(),
        serde_json::json!(pv.n_test_embeddings),
    );
    obj.insert(
        "n_frames_below_reset".to_string(),
        serde_json::json!(pv.n_frames_below_reset),
    );
    obj.insert(
        "agc_converged".to_string(),
        serde_json::json!(pv.agc_converged),
    );
    obj.insert(
        "vad_speech_frames".to_string(),
        serde_json::json!(pv.vad_speech_frames),
    );
    obj.insert(
        "per_frame_scores".to_string(),
        serde_json::json!(pv.per_frame_scores),
    );
    obj.insert(
        "verifier_score_trajectory".to_string(),
        serde_json::json!(pv.verifier_score_trajectory),
    );
    obj.insert(
        "warmup_completed".to_string(),
        serde_json::json!(pv.warmup_completed),
    );
    obj.insert(
        "warmup_n_embeddings".to_string(),
        serde_json::json!(pv.warmup_n_embeddings),
    );
    obj.insert(
        "warmup_max_rolling_sum".to_string(),
        serde_json::json!(pv.warmup_max_rolling_sum),
    );
    obj.insert(
        "warmup_peak_verifier_score".to_string(),
        serde_json::json!(pv.warmup_peak_verifier_score),
    );
    obj.insert(
        "n_warmup_suppressed_frames".to_string(),
        serde_json::json!(pv.n_warmup_suppressed_frames),
    );
    obj.insert(
        "candidates_created".to_string(),
        serde_json::json!(pv.candidates_created),
    );
    obj.insert(
        "candidates_confirmed".to_string(),
        serde_json::json!(pv.candidates_confirmed),
    );
    obj.insert(
        "candidates_expired".to_string(),
        serde_json::json!(pv.candidates_expired),
    );
    obj.insert(
        "first_trigger_frame_idx".to_string(),
        serde_json::json!(pv.first_trigger_frame_idx),
    );
    obj.insert(
        "verifier_threshold".to_string(),
        serde_json::json!(pv.verifier_threshold),
    );
    obj.insert(
        "verifier_trained".to_string(),
        serde_json::json!(pv.verifier_trained),
    );
    obj.insert("latency_ms".to_string(), serde_json::json!(pv.latency_ms));
    // Positive-only miss evidence (mahbot-1005 §2/§3/§4).
    if category.is_none() {
        let verdict = classify_miss(pv);
        obj.insert("verdict".to_string(), serde_json::json!(verdict.as_str()));
        if pv.verifier_trained && !pv.detected {
            obj.insert(
                "verifier_rejection_margin".to_string(),
                serde_json::json!((pv.verifier_threshold - pv.verifier_score).max(0.0)),
            );
        }
        // Trigger-point evidence (mahbot-1005 §3): the classifier's score at the
        // first trigger frame and where in the utterance it occurred.  VAD is
        // binary (is_speech), so "noise burst" vs "silence" at the trigger
        // cannot be distinguished by VAD alone — the frame's total_score vs
        // NO_MATCH_RESET_THRESHOLD is the closest signal and is reported raw.
        if let Some(idx) = pv.first_trigger_frame_idx
            && let Some(frame) = pv.per_frame_scores.get(idx)
        {
            let frac = if pv.per_frame_scores.is_empty() {
                0.0
            } else {
                idx as f64 / pv.per_frame_scores.len() as f64
            };
            obj.insert(
                "trigger_frame_total_score".to_string(),
                serde_json::json!(frame[0]),
            );
            obj.insert(
                "trigger_frame_rolling_sum".to_string(),
                serde_json::json!(frame[ROLLING_SUM_IDX]),
            );
            obj.insert(
                "trigger_frame_effective_threshold".to_string(),
                serde_json::json!(frame[THRESHOLD_IDX]),
            );
            obj.insert("trigger_position_frac".to_string(), serde_json::json!(frac));
            obj.insert(
                "trigger_frame_kind".to_string(),
                serde_json::json!(if frac <= 0.25 {
                    "speech_onset"
                } else {
                    "sustained_speech"
                }),
            );
        }
        // "never evaluated" vs "rejected with zero confidence" (mahbot-1005 §4):
        // a verifier_score of 0.0 means the verifier was never evaluated when
        // the warm-up was incomplete or the verifier is untrained; otherwise it
        // is a genuine zero-confidence evaluation.
        if pv.verifier_score == 0.0 {
            let meaning =
                if pv.verifier_trained && pv.warmup_completed && !pv.per_frame_scores.is_empty() {
                    "evaluated_zero_confidence"
                } else {
                    "never_evaluated"
                };
            obj.insert(
                "verifier_score_meaning".to_string(),
                serde_json::json!(meaning),
            );
        }
    }
    serde_json::Value::Object(obj)
}

// ── Distribution helpers (mahbot-1005 §3/§4) ──────────────────────────────

/// Decile boundaries of a score distribution (10 values, min→max inclusive of
/// each 10% quantile).  Returns `None` for empty input (mahbot-1005 §3).
fn deciles(scores: &[f32]) -> Option<[f32; 10]> {
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

/// Fixed-range histogram with `n_buckets` buckets over `[lo, hi]`.  Values
/// below `lo` fall into bucket 0, values above `hi` into the last bucket.
fn histogram_buckets(scores: &[f32], n_buckets: usize, lo: f32, hi: f32) -> Vec<usize> {
    let mut buckets = vec![0usize; n_buckets.max(1)];
    if scores.is_empty() || n_buckets == 0 {
        return buckets;
    }
    let span = (hi - lo).max(f32::EPSILON);
    for &s in scores {
        let t = ((s - lo) / span * n_buckets as f32).clamp(0.0, n_buckets as f32 - 1.0);
        buckets[t as usize] += 1;
    }
    buckets
}

/// Identify the PCM augmentation type from a variant label (mahbot-1005 §6).
/// Labels are produced by [`pcm_augment_enrollment_variants`] with one of five
/// suffixes: `_original`, `_speed_down`, `_speed_up`, `_vol_down`, `_noise`.
fn augmentation_type(label: &str) -> &'static str {
    if label.ends_with("_speed_down") {
        "speed_down"
    } else if label.ends_with("_speed_up") {
        "speed_up"
    } else if label.ends_with("_vol_down") {
        "vol_down"
    } else if label.ends_with("_noise") {
        "noise"
    } else if label.ends_with("_original") {
        "original"
    } else {
        "other"
    }
}

/// L2-normalized mean embedding (centroid) of the given sequences' embeddings.
/// Returns `None` when no embeddings exist (mahbot-1005 §6).
fn embedding_centroid(
    sequences: &[crate::audio::embedding_sequence::EmbeddingSequence],
) -> Option<Vec<f32>> {
    let mut n = 0usize;
    let mut sum: Option<Vec<f32>> = None;
    for seq in sequences {
        for emb in &seq.embeddings {
            match &mut sum {
                Some(s) => {
                    debug_assert_eq!(s.len(), emb.len());
                    for (a, b) in s.iter_mut().zip(emb) {
                        *a += b;
                    }
                }
                None => sum = Some(emb.clone()),
            }
            n += 1;
        }
    }
    let mut centroid = sum?;
    for v in &mut centroid {
        *v /= n as f32;
    }
    let norm = centroid
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt()
        .max(1e-10);
    for v in &mut centroid {
        *v /= norm;
    }
    Some(centroid)
}

/// Cosine similarity between two equal-length vectors (L2-normalizes on the fly).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let na = a.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
    let nb = b.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
    a.iter().zip(b).map(|(x, y)| x * y / (na * nb)).sum::<f32>()
}

/// Process a list of audio clips through the detection pipeline, recording
/// results in `metrics`.  Shared helper for positive and negative detection
/// test blocks, eliminating the repetitive match-and-track boilerplate.
///
/// # Parameters
/// - `variants`: audio clips with descriptive labels.
/// - `classifier`, `verifier`: trained models passed to the streaming pipeline.
/// - `metrics`: records total/detected; `on_detection` fills detected or
///   false_accepts.
/// - `on_detection`: called with `(&mut metrics, label_str)` when the
///   wake word is detected (for positives: increment `.detected`; for
///   negatives: push to `.false_accepts`).
fn test_detection_samples(
    variants: &[(Vec<f32>, String)],
    classifier: &WakeWordClassifier,
    verifier: &VoiceVerifier,
    metrics: &mut DetectionMetrics,
    on_detection: impl Fn(&mut DetectionMetrics, &str),
    mut adaptive_state: Option<&mut super::AdaptiveThresholdState>,
    cold_start: bool,
) {
    // Set classifier + verifier in global state for the streaming pipeline.
    // score_stride8_window reads these from voice_state().
    super::set_classifier_weights(classifier.weights_ref().clone());
    super::set_verifier(verifier.clone());

    for (i, (samples, label)) in variants.iter().enumerate() {
        info!(
            "  Variant {}/{}: {label} — processing ({})",
            i + 1,
            variants.len(),
            if cold_start { "cold start" } else { "warm" }
        );
        metrics.total += 1;
        let mut ctx = super::PipelineCtx::new();
        // Clone the shared adaptive state into this variant's ctx so the
        // adaptive threshold is active from the first frame, simulating a
        // continuous pipeline (mahbot-845, reviewer_3).  Without this the
        // adaptive state never exits its 5-frame bootstrap because each
        // variant gets a fresh ctx, keeping all benchmark metrics measured
        // against the static threshold.
        //
        // Cold start (mahbot-1006 D/F): production bootstraps the adaptive
        // threshold from the first 5 real per-frame scores of actual background
        // audio after each segment boundary.  The cold pass therefore uses a
        // fresh AdaptiveThresholdState::new() per variant (no shared-state
        // propagation) so the threshold trajectory matches production exactly.
        if !cold_start && let Some(ref mut state) = adaptive_state {
            ctx.adaptive_threshold = state.clone();
        }
        // Consume verifier warm-up before the test utterance so the latency
        // timer measures only the wake word (mahbot-922, reviewer feedback).
        //
        // Cold start (mahbot-1006 D): production has no warm-up — the first
        // VERIFIER_WARMUP_EMBEDDINGS embeddings of every utterance after
        // silence are suppressed during verifier warm-up.  The cold pass skips
        // consume_warmup so the test utterance itself consumes the warm-up
        // period, exactly like production's post-silence start.
        let (warmup_n_embeddings, warmup_completed) = if cold_start {
            (0, false)
        } else {
            consume_warmup(&mut ctx);
            let n = ctx.embedding_ring.len();
            let completed = n >= crate::audio::voice_verifier::VERIFIER_WARMUP_EMBEDDINGS;
            info!(
                "  Variant {}/{}: {label} — warm-up {} ({} embeddings)",
                i + 1,
                variants.len(),
                if completed {
                    "completed ✓"
                } else {
                    "FAILED ✗"
                },
                n,
            );
            (n, completed)
        };
        let result = run_streaming_detection(samples, &mut ctx);
        // Propagate the updated adaptive state for the next variant (warm pass
        // only — the cold pass keeps each variant's bootstrap independent).
        //
        // When the trailing silence fires a segment boundary (`!detected` AND
        // the ring was cleared — reset_detection_segment is the only
        // non-detection path that empties the warm pass's ring), the boundary
        // resets ctx.adaptive_threshold back to bootstrap.  Propagating that
        // bootstrap state would defeat the shared-state design (mahbot-845)
        // which keeps the adaptive code path active across variants to avoid
        // measuring everything against the static threshold — and with item
        // H's extended flush every non-detecting variant fires the boundary.
        // Instead, carry the pre-flush snapshot captured at the end of the
        // utterance so the shared state keeps adapting to each variant's
        // frames.  The cold pass measures the bootstrap behavior separately
        // (mahbot-1006 F).
        if !cold_start && let Some(ref mut state) = adaptive_state {
            let boundary_fired = !result.detected && ctx.embedding_ring.is_empty();
            if boundary_fired {
                **state = result.adaptive_state_pre_flush.clone();
            } else {
                **state = ctx.adaptive_threshold.clone();
            }
        }
        let peak = ctx.instrumentation.peak_score;
        if result.detected {
            if let Some(lat) = result.latency_ms {
                metrics.latencies.push(lat);
            }
            on_detection(metrics, label);
        }
        // Record per-variant result (mahbot-845) with verifier score (mahbot-859)
        // and per-variant instrumentation (mahbot-886, mahbot-1005).
        metrics.per_variant.push(build_per_variant_result(
            label,
            &result,
            &ctx,
            warmup_completed,
            warmup_n_embeddings,
            verifier,
        ));
        info!(
            "  Variant {}/{}: {label} — {} (peak_score={:.4})",
            i + 1,
            variants.len(),
            if result.detected {
                "DETECTED"
            } else {
                "passed"
            },
            peak,
        );
    }
}

// ── Volume sweep (mahbot-844 Part 3, informational) ──────────────────────

/// Apply a gain in dB to PCM audio and optionally hard-clip.
///
/// Uses the canonical [`crate::util::apply_gain`] internally, then
/// optionally hard-clips to [-1.0, 1.0] for volume-sweep testing.
fn apply_gain(samples: &[f32], gain_db: f32, hard_clip: bool) -> Vec<f32> {
    let amplified = crate::util::apply_gain(samples, gain_db);
    if hard_clip {
        amplified.iter().map(|&s| s.clamp(-1.0, 1.0)).collect()
    } else {
        amplified
    }
}

/// Run an informational volume sweep on positive detection variants.
///
/// Reports detection rate per level plus per-variant peak-score distributions
/// so the sweep shows not just pass/fail but HOW scores degrade with volume
/// (mahbot-1005 §8).  Not gated — results are reported in the JSON output.
fn run_volume_sweep(
    positive_variants: &[(Vec<f32>, String)],
    classifier: &WakeWordClassifier,
    verifier: &VoiceVerifier,
) -> Vec<(&'static str, f64, Vec<f32>, Vec<f32>)> {
    let mut results = Vec::new();
    for (label, gain_db, hard_clip) in VOLUME_SWEEP_LEVELS {
        let mut metrics = DetectionMetrics::default();
        let processed: Vec<(Vec<f32>, String)> = positive_variants
            .iter()
            .map(|(pcm, l)| {
                let adjusted = apply_gain(pcm, *gain_db, *hard_clip);
                (adjusted, l.clone())
            })
            .collect();
        test_detection_samples(
            &processed,
            classifier,
            verifier,
            &mut metrics,
            |m, _| {
                m.detected += 1;
            },
            None,
            false, // volume sweep uses the warm pass only (informational phase)
        );
        let rate = metrics.detection_rate();
        let peak_scores: Vec<f32> = metrics.per_variant.iter().map(|pv| pv.peak_score).collect();
        let rolling_sums: Vec<f32> = metrics
            .per_variant
            .iter()
            .map(|pv| pv.max_rolling_sum)
            .collect();
        info!("Volume sweep {label} ({gain_db}dB): {:.1}%", rate * 100.0);
        results.push((*label, rate, peak_scores, rolling_sums));
    }
    results
}

// ── Mid-utterance detection test (mahbot-844 Part 4, informational) ───────

/// Run an informational mid-utterance detection test.
///
/// Slices already-synthesized speech (confusable/unrelated) before wake word
/// audio and tests whether the wake word is still detected.
///
/// `confusable_variants` **must** contain only hard-tier confusable variants
/// (see mahbot-871 — the mid-utterance test always uses the hardest confusable
/// distractor regardless of the selected benchmark tier).
fn run_mid_utterance_test(
    pos_test_variants: &[(Vec<f32>, String)],
    confusable_variants: &[(Vec<f32>, String)],
    unrelated_variants: &[(Vec<f32>, String)],
) -> Vec<(&'static str, bool)> {
    let gap_silence = |secs: f64| -> Vec<f32> {
        let samples = (secs * f64::from(TARGET_SAMPLE_RATE)) as usize;
        vec![0.0f32; samples]
    };

    let mut results = Vec::new();

    // 1. Immediate transition: confusable prelude → wake word with no gap
    if let (Some((conf_pcm, _)), Some((wake_pcm, _))) =
        (confusable_variants.first(), pos_test_variants.first())
    {
        let mut spliced = conf_pcm.clone();
        spliced.extend_from_slice(wake_pcm);
        let mut ctx = super::PipelineCtx::new();
        consume_warmup(&mut ctx);
        let result = run_streaming_detection(&spliced, &mut ctx);
        let detected = result.detected;
        info!("Mid-utterance 'immediate_transition': detected={detected}");
        results.push(("immediate_transition", detected));
    }

    // 2. Brief gap (50ms): confusable prelude → 50ms silence → wake word
    if let (Some((conf_pcm, _)), Some((wake_pcm, _))) =
        (confusable_variants.first(), pos_test_variants.first())
    {
        let mut spliced = conf_pcm.clone();
        spliced.extend_from_slice(&gap_silence(0.05));
        spliced.extend_from_slice(wake_pcm);
        let mut ctx = super::PipelineCtx::new();
        consume_warmup(&mut ctx);
        let result = run_streaming_detection(&spliced, &mut ctx);
        let detected = result.detected;
        info!("Mid-utterance 'brief_gap_50ms': detected={detected}");
        results.push(("brief_gap_50ms", detected));
    }

    // 3. Long prelude (~6s): unrelated prelude → 50ms gap → wake word
    if let (Some((unrel_pcm, _)), Some((wake_pcm, _))) =
        (unrelated_variants.first(), pos_test_variants.first())
    {
        let mut spliced = unrel_pcm.clone();
        spliced.extend_from_slice(&gap_silence(0.05));
        spliced.extend_from_slice(wake_pcm);
        let mut ctx = super::PipelineCtx::new();
        consume_warmup(&mut ctx);
        let result = run_streaming_detection(&spliced, &mut ctx);
        let detected = result.detected;
        info!("Mid-utterance 'long_prelude_6s': detected={detected}");
        results.push(("long_prelude_6s", detected));
    }

    // 4. Confusable prelude ("hey max"): splice "hey max" → wake word
    // "hey max" is the first confusable phrase that phonetically resembles
    // "hey mahbot".
    if let (Some((max_pcm, _)), Some((wake_pcm, _))) =
        (confusable_variants.first(), pos_test_variants.first())
    {
        let mut spliced = max_pcm.clone();
        spliced.extend_from_slice(&gap_silence(0.05));
        spliced.extend_from_slice(wake_pcm);
        let mut ctx = super::PipelineCtx::new();
        consume_warmup(&mut ctx);
        let result = run_streaming_detection(&spliced, &mut ctx);
        let detected = result.detected;
        info!("Mid-utterance 'confusable_prelude': detected={detected}");
        results.push(("confusable_prelude", detected));
    }

    results
}

// ── Noise-overlapped detection test (mahbot-845) ─────────────────────────

/// Compute RMS (root mean square) of a PCM audio buffer.
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Mix speech and noise at a given SNR (in dB).
///
/// `speech` and `noise` should be the same length.  The noise is scaled so
/// that `SNR = 20 * log10(rms_speech / (scale * rms_noise))`.
/// When `snr_db` is `f32::INFINITY`, returns the speech unchanged.
fn mix_at_snr(speech: &[f32], noise: &[f32], snr_db: f32) -> Vec<f32> {
    if !snr_db.is_finite() {
        return speech.to_vec();
    }
    let speech_rms = rms(speech);
    let noise_rms = rms(noise);
    if noise_rms < 1e-10 || speech_rms < 1e-10 {
        return speech.to_vec();
    }
    // SNR = 20 * log10(speech_rms / (scale * noise_rms))
    // → scale = speech_rms / (noise_rms * 10^(snr_db / 20))
    let target_ratio = 10.0_f32.powf(snr_db / 20.0);
    let scale = speech_rms / (noise_rms * target_ratio);

    let min_len = speech.len().min(noise.len());
    let mut mixed = Vec::with_capacity(min_len);
    for i in 0..min_len {
        mixed.push(speech[i] + noise[i] * scale);
    }
    // If speech is longer, append remaining speech (noise exhausted)
    if min_len < speech.len() {
        mixed.extend_from_slice(&speech[min_len..]);
    }
    mixed
}

/// Run noise-overlapped detection tests.
///
/// For each combination of SNR level and noise type, mix the wake word
/// variants with the noise and test detection.  Returns `(key, rate,
/// per_variant_detail)` per combination so per-variant peak scores and verifier
/// evidence are available for false-reject root-causing (mahbot-1005 §8).
fn run_noise_overlap_test(
    positive_variants: &[(Vec<f32>, String)],
    classifier: &WakeWordClassifier,
    verifier: &VoiceVerifier,
) -> Vec<(String, f64, Vec<serde_json::Value>)> {
    // Set classifier + verifier in global state (test_detection_samples does
    // this too but we inline the loop to share adaptive state across variants).
    super::set_classifier_weights(classifier.weights_ref().clone());
    super::set_verifier(verifier.clone());

    // Pre-warm a shared adaptive threshold state so the benchmark actually
    // exercises the adaptive code path (reviewer_2, mahbot-845).  Without
    // this, test_detection_samples — which creates a fresh PipelineCtx per
    // variant — never exits the 5-frame bootstrap, so all measurements use
    // the static threshold.
    let mut shared_adaptive = super::AdaptiveThresholdState::warmed();

    let mut results = Vec::new();

    for (snr_label, snr_db) in NOISE_OVERLAP_SNRS {
        for (noise_label, noise_gen) in NOISE_OVERLAP_TYPES {
            let noise = noise_gen();
            let mut metrics = DetectionMetrics::default();

            for (pcm, label) in positive_variants {
                let mixed_pcm = mix_at_snr(pcm, &noise, *snr_db);
                metrics.total += 1;

                // Each variant gets a fresh ctx (clean score_window,
                // embedding_ring, etc.) but carries the shared adaptive
                // threshold state forward across detection attempts.
                let mut ctx = super::PipelineCtx::new();
                ctx.adaptive_threshold = shared_adaptive.clone();
                // adaptive_k is already set by PipelineCtx::new() from config.

                // Consume verifier warm-up so the latency timer starts at the
                // noise-overlapped utterance (mahbot-922).
                consume_warmup(&mut ctx);
                let warmup_n_embeddings = ctx.embedding_ring.len();
                let warmup_completed =
                    warmup_n_embeddings >= crate::audio::voice_verifier::VERIFIER_WARMUP_EMBEDDINGS;
                let result = run_streaming_detection(&mixed_pcm, &mut ctx);

                // Persist the updated adaptive state for the next variant,
                // with the same boundary-fire guard as test_detection_samples
                // (mahbot-1006 F): when the trailing silence fired a segment
                // boundary (`!detected` AND the ring was cleared), the
                // boundary reset ctx.adaptive_threshold to bootstrap.
                // Propagating that bootstrap state would defeat the
                // shared-state design (mahbot-845) — and with item H's
                // extended flush every non-detecting variant fires the
                // boundary, so the phase would otherwise re-bootstrap per
                // variant and measure mostly against the static threshold.
                // Carry the pre-flush snapshot instead so the shared state
                // keeps adapting to each variant's frames.
                let boundary_fired = !result.detected && ctx.embedding_ring.is_empty();
                shared_adaptive = if boundary_fired {
                    result.adaptive_state_pre_flush.clone()
                } else {
                    ctx.adaptive_threshold.clone()
                };

                if result.detected {
                    if let Some(lat) = result.latency_ms {
                        metrics.latencies.push(lat);
                    }
                    metrics.detected += 1;
                }
                metrics.per_variant.push(build_per_variant_result(
                    label,
                    &result,
                    &ctx,
                    warmup_completed,
                    warmup_n_embeddings,
                    verifier,
                ));
            }

            let rate = metrics.detection_rate();
            let key = format!("{snr_label}_{noise_label}");
            let detail: Vec<serde_json::Value> = metrics
                .per_variant
                .iter()
                .map(|pv| pv_to_json(pv, Some("noise_overlap")))
                .collect();
            info!(
                "Noise overlap {snr_label} / {noise_label}: {:.1}% detection ({}/{})",
                rate * 100.0,
                metrics.detected,
                metrics.total,
            );
            results.push((key, rate, detail));
        }
    }

    results
}

// ═══════════════════════════════════════════════════════════════════════════
// The main integration test / benchmark entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Run the full voice pipeline E2E benchmark.
///
/// This is called from `voice::run_voice_pipeline_benchmark()` which is the
/// entry point for `benches/voice_pipeline_e2e.rs`.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::if_not_else
)]
pub(crate) fn run_internal() {
    // ── Phase index constants for type-safe timing access ────────────────
    // Must be before any statements (clippy items_after_statements).
    const P_ENROLLMENT_AUDIO: usize = 0;
    const P_VAD_ENROLLMENT: usize = 1;
    const P_NEG_TRAINING_DATA: usize = 2;
    const P_CLASSIFIER_TRAINING: usize = 3;
    const P_VERIFIER_TRAINING: usize = 4;
    const P_GLOBAL_STATE: usize = 5;
    const P_STREAMING_SETUP: usize = 6;
    const P_POSITIVE_VARIANTS: usize = 7;
    const P_CONFUSABLE_NEGATIVES: usize = 8;
    const P_UNRELATED_NEGATIVES: usize = 9;
    const P_SILENCE_NEGATIVES: usize = 10;
    const P_NOISE_PROFILES: usize = 11;
    const P_COOLDOWN: usize = 12;
    const P_NOISE_OVERLAP: usize = 13;
    const P_TEARDOWN: usize = 14;
    const NUM_PHASES: usize = 15;

    // ── Heartbeat drop guard (mahbot-944) ──────────────────────────────
    // Defined as an item (before any statements) so it satisfies Clippy's
    // items_after_statements requirement.  Used by the heartbeat thread to
    // ensure the stop flag is set even on panic unwind.
    struct HeartbeatGuard(std::sync::Arc<std::sync::atomic::AtomicBool>);

    impl Drop for HeartbeatGuard {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    // Initialize a tracing subscriber so progress info!() messages appear
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .parse("info")
                .expect("info env filter"),
        )
        .try_init();

    // ── Phase timing ─────────────────────────────────────────────────
    // We measure each named phase with Instant timestamps.  Phase 13's
    // cooldown sleep is excluded from its timing.
    let mut phase_starts: Vec<(&str, Instant)> = Vec::new();
    macro_rules! phase_start {
        ($name:literal) => {{
            // Both stderr (for live visibility) and info! (for tracing/logs.db).
            // Stderr ensures output is visible even if the tracing subscriber is
            // not initialized or buffers messages (mahbot-944).
            eprintln!("─── {}. {} ───", phase_starts.len() + 1, $name);
            info!("─── {}. {} ───", phase_starts.len() + 1, $name);
            phase_starts.push(($name, Instant::now()));
        }};
    }
    macro_rules! phase_end_ms {
        () => {{
            let (name, start) = phase_starts.last().unwrap();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            eprintln!("  → {} completed in {:.0}ms", name, elapsed);
            info!("  → {} completed in {:.0}ms", name, elapsed);
            elapsed as u64
        }};
    }

    let overall_start = Instant::now();
    let mut phase_times = [0u64; NUM_PHASES];

    info!("═══ Voice Pipeline E2E Benchmark ═══");

    info!("Benchmark tier: all three tiers tested (Easy FA≤0, Medium FA≤1, Hard FA≤2)");

    // ── Heartbeat thread (mahbot-944) ──────────────────────────────────
    // Prints a pulse every 60 s so the operator can confirm the benchmark
    // is alive and making progress (vs. parked threads from nested runtime
    // hangs).
    //
    // Uses 1 s sleep intervals so the thread responds to the stop flag
    // within 1 s (join latency).  A counter gates the print cadence — the
    // first pulse prints immediately so a hang in the first 60 s is still
    // detectable.
    //
    // A drop guard (`_heartbeat_guard`) sets the stop flag on panic unwind,
    // ensuring the heartbeat thread exits promptly even when the function
    // panics before the explicit stop-then-join at the end.
    let heartbeat_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let _heartbeat_guard = HeartbeatGuard(heartbeat_stop.clone());
    let heartbeat_handle = {
        let stop = heartbeat_stop.clone();
        let start = overall_start;
        std::thread::spawn(move || {
            let mut counter: u64 = 0;
            loop {
                // Print immediately on first iteration (counter=0, 0 % 60 = 0),
                // then every 60 ticks (~60 s with 1 s sleep between ticks).
                if counter.is_multiple_of(60) {
                    let elapsed = start.elapsed();
                    eprintln!(
                        "[heartbeat] benchmark still running — elapsed: {}m{:02}s",
                        elapsed.as_secs() / 60,
                        elapsed.as_secs() % 60,
                    );
                }
                std::thread::sleep(Duration::from_secs(1));
                counter += 1;
                if stop.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
            }
        })
    };

    // ── 0. Initialize global state ─────────────────────────────────────
    // Set CONFIG storage root so model paths resolve.
    if crate::config::CONFIG.try_storage_root().is_none() {
        let mahbot_dir = crate::config::default_config_dir()
            .expect("Cannot resolve home directory for ~/.mahbot");
        crate::config::CONFIG.set_storage_root(mahbot_dir.clone());
        info!("CONFIG storage root set to: {}", mahbot_dir.display());
    }

    // Determine cache settings
    let cache_dir_path = cache_dir();
    if let Err(e) = std::fs::create_dir_all(&cache_dir_path) {
        eprintln!(
            "WARNING: Cannot create cache directory {}: {e}",
            cache_dir_path.display()
        );
    }

    // Compute model version hash for TTS caching (from compile-time constants)
    let model_version_hash = super::tts_model_version_hash();
    info!("TTS model version hash: {}", &model_version_hash[..16]);

    // Initialize TTS module
    crate::audio::tts::init_global()
        .unwrap_or_else(|e| warn!("tts::init_global() already called: {e}"));

    // Ensure voice pipeline state is initialized
    super::init_global().unwrap_or_else(|e| warn!("voice::init_global() already called: {e}"));

    // ── Prerequisites ───────────────────────────────────────────────
    if let Err(msg) = crate::audio::tts::ensure_ready() {
        panic!("{msg}\nRun the application first to download TTS models (~400 MB).");
    }

    if let Err(msg) = ensure_voice_models_loaded() {
        panic!("{msg}\nRun the application first to download voice models.");
    }

    let available_styles = tts::list_voice_styles();
    info!(
        "TTS ready with {} voice styles: {:?}",
        available_styles.len(),
        available_styles
    );

    // ── Phase 1: Generate enrollment audio ───────────────────────────────
    phase_start!("Phase 1: Generating enrollment audio");
    let enrollment_variants = generate_enrollment_variants_cached(
        &available_styles,
        &model_version_hash,
        &cache_dir_path,
    );
    if enrollment_variants.is_empty() {
        warn!(
            "Need at least one enrollment variant. TTS synthesis may have failed for all styles."
        );
        eprintln!(
            "FATAL: Need at least one enrollment variant. TTS synthesis may have failed for all styles."
        );
        return;
    }
    info!(
        "Generated {} enrollment variants",
        enrollment_variants.len()
    );

    // ── Train/test split at the CLIP level (mahbot-932 Fix 5, mahbot-1006 B/I) ──
    // The split strategy is unchanged (2/3 : 1/3 by enrollment-list order,
    // tail = test).  It now happens on the raw enrollment clips BEFORE AGC/VAD/
    // augmentation because the augmentation itself moved inside the per-clip
    // processing (production order: AGC → VAD → augment; mahbot-1006 B/I).
    // Splitting the clips (rather than the 5× augmented list) also removes the
    // previous leakage where augmented variants of the same raw clip spanned
    // the train/test boundary.
    //
    // - Train clips → vad_segment_and_enroll (AGC → VAD → augment → embeddings).
    // - Test clips → raw-level PCM augmentation (original + speed/volume/noise
    //   applied to the raw TTS audio), fed through the streaming detection path
    //   which applies AGC exactly like production's live mic loop.
    let n_train_clips = (enrollment_variants.len() * 2 / 3).max(1);
    let mut all_clips = enrollment_variants;
    let test_clips = all_clips.split_off(n_train_clips);
    let train_clips = all_clips;

    // Test-set raw-level PCM augmentation (mahbot-932 Fix 5 semantics kept:
    // the detection test covers all 5 augmentation types at the raw level).
    let test_variants = pcm_augment_enrollment_variants(&test_clips);
    info!(
        "PCM augmentation (test clips): {} clips -> {} raw-level test variants (original + \
         speed-down + speed-up + vol-down + noise per clip)",
        test_clips.len(),
        test_variants.len(),
    );
    phase_times[P_ENROLLMENT_AUDIO] = phase_end_ms!();

    // ── Phase 2: VAD-gated enrollment (train clips) ─────────────────────
    phase_start!("Phase 2: VAD-gated enrollment");

    info!(
        "Train/test split (clip level): {} train clips, {} test clips (mahbot-932)",
        train_clips.len(),
        test_clips.len()
    );

    let pos_sequences = vad_segment_and_enroll(&train_clips);
    if pos_sequences.is_empty() {
        warn!(
            "VAD-gated enrollment produced no utterances from {} training clips",
            train_clips.len(),
        );
        eprintln!(
            "FATAL: VAD-gated enrollment produced no utterances from {} training clips",
            train_clips.len(),
        );
        return;
    }
    info!(
        "VAD-gated enrollment: {} dense embeddings from {} sequences",
        pos_sequences
            .iter()
            .map(|s| s.embeddings.len())
            .sum::<usize>(),
        pos_sequences.len(),
    );

    phase_times[P_VAD_ENROLLMENT] = phase_end_ms!();

    // ── Phase 3: Generate negative training data ─────────────────────────
    phase_start!("Phase 3: Generating negative training data");

    // Pre-warm production caches for confusable and unrelated embeddings
    // (mahbot-880).  These are populated by `prewarm_*` during normal app
    // startup, but the benchmark runs synchronously and never calls prewarm.
    // We call it here so the benchmark uses the same cached embeddings as
    // production, ensuring benchmark results reflect real behavior.
    info!("Pre-warming production negative embedding caches (mahbot-880)...");
    {
        // Use the outer runtime's handle instead of creating a nested runtime
        // (mahbot-944).  Creating a second tokio runtime inside spawn_blocking
        // causes BlockingPool::shutdown to hang when the inner runtime drops,
        // because the oneshot stop signal races with blocking thread idle
        // timeouts.  Handle::current().block_on() runs the prewarm futures on
        // the outer runtime's timer and blocking pool, avoiding the nested
        // runtime entirely.
        let handle = tokio::runtime::Handle::current();
        handle.block_on(super::prewarm_confusable_embeddings());
        handle.block_on(super::prewarm_unrelated_embeddings());
    }

    // Check if production caches are populated.  After mahbot-923 the pipeline
    // uses dense stride-8 embeddings throughout — no streaming cache needed.
    let confusable_dense_cache = super::confusable_dense_embeddings();
    let unrelated_dense_cache = super::unrelated_dense_embeddings();

    // Post-prewarm assertion: production caches must be non-empty.
    assert!(
        !confusable_dense_cache.is_empty(),
        "Confusable dense embeddings cache is empty after prewarm. \
         voice::prewarm_confusable_embeddings() did not populate the cache."
    );
    assert!(
        !unrelated_dense_cache.is_empty(),
        "Unrelated dense embeddings cache is empty after prewarm. \
         voice::prewarm_unrelated_embeddings() did not populate the cache."
    );

    let (classifier_neg_sequences, verifier_neg_sequences, per_negative_sequence_weights);

    // Generate ambient noise sequences (mahbot-932 Fix 7) — replaces synthetic negatives.
    let ambient_sequences = generate_ambient_noise_sequences();
    info!(
        "Generated {} ambient noise sequences from {} noise profiles × 2 SNR levels (mahbot-932)",
        ambient_sequences.len(),
        NOISE_PROFILES.len(),
    );

    // ── Production cache path (mahbot-880, mahbot-923) ──────────────────
    // Use the shared OnceLock caches populated by the prewarm functions
    // above.  After mahbot-923 the pipeline uses dense stride-8 embeddings
    // throughout — both classifier and verifier use the same dense caches.
    // Ambient + owner-negative replace the old synthetic negatives (mahbot-932).
    info!(
        "Using production pre-warmed caches: {} confusable + {} unrelated dense (mahbot-880, mahbot-923)",
        confusable_dense_cache.len(),
        unrelated_dense_cache.len(),
    );

    // Build classifier negative sequences: ambient → owner → confusable → unrelated
    // (mahbot-932 Fix 7, Fix 8).  No synthetic negatives — production uses zero
    // synthetic embedding-space negatives.
    let mut classifier_neg_seqs: Vec<EmbeddingSequence> = Vec::new();
    classifier_neg_seqs.extend(ambient_sequences.clone());

    // Owner-negative sequences
    let owner_seqs =
        generate_owner_negative_sequences(&available_styles, &model_version_hash, &cache_dir_path);
    info!(
        "Generated {} owner-negative sequences (mahbot-932 Fix 6)",
        owner_seqs.len(),
    );
    classifier_neg_seqs.extend(owner_seqs.clone());
    classifier_neg_seqs.extend_from_slice(confusable_dense_cache);
    classifier_neg_seqs.extend_from_slice(unrelated_dense_cache);

    let n_dense_total = classifier_neg_seqs.len();
    classifier_neg_sequences = classifier_neg_seqs;

    // Verifier uses the same dense cache embeddings as the classifier.
    // Build verifier negative sequences in 4-tier order: ambient → owner → unrelated → confusable
    let mut verifier_neg_seqs: Vec<EmbeddingSequence> = Vec::new();
    verifier_neg_seqs.extend(ambient_sequences);
    verifier_neg_seqs.extend(owner_seqs);
    verifier_neg_seqs.extend_from_slice(unrelated_dense_cache);
    verifier_neg_seqs.extend_from_slice(confusable_dense_cache);
    verifier_neg_sequences = verifier_neg_seqs;

    // Build per-sequence 4-tier weights matching production (mahbot-932 Fix 8):
    // ambient (1.0×) → owner-negative (OWNER_NEGATIVE_UPWEIGHT×)
    // → unrelated (UNRELATED_UPWEIGHT×) → confusable (CONFUSABLE_UPWEIGHT×).
    let n_ambient = classifier_neg_sequences
        .iter()
        .filter(|s| s.source == Source::Ambient)
        .count();
    let n_owner = classifier_neg_sequences
        .iter()
        .filter(|s| s.source == Source::Owner)
        .count();
    let n_confusable = confusable_dense_cache.len();
    let n_unrelated = unrelated_dense_cache.len();
    let n_seq_total = n_ambient + n_owner + n_confusable + n_unrelated;

    let mut pw: Vec<f32> = Vec::with_capacity(n_seq_total);
    pw.extend(std::iter::repeat_n(1.0, n_ambient));
    pw.extend(std::iter::repeat_n(
        crate::audio::voice_verifier::OWNER_NEGATIVE_UPWEIGHT,
        n_owner,
    ));
    pw.extend(std::iter::repeat_n(
        crate::audio::voice_verifier::UNRELATED_UPWEIGHT,
        n_unrelated,
    ));
    pw.extend(std::iter::repeat_n(
        crate::audio::voice_verifier::CONFUSABLE_UPWEIGHT,
        n_confusable,
    ));
    per_negative_sequence_weights = pw;

    // Structural guard: verify weight tier boundaries (mahbot-880).
    crate::audio::voice_verifier::assert_weight_tier(
        &per_negative_sequence_weights,
        0,
        n_ambient,
        1.0,
        "ambient",
    );
    crate::audio::voice_verifier::assert_weight_tier(
        &per_negative_sequence_weights,
        n_ambient,
        n_owner,
        crate::audio::voice_verifier::OWNER_NEGATIVE_UPWEIGHT,
        "owner-negative",
    );
    crate::audio::voice_verifier::assert_weight_tier(
        &per_negative_sequence_weights,
        n_ambient + n_owner,
        n_unrelated,
        crate::audio::voice_verifier::UNRELATED_UPWEIGHT,
        "unrelated",
    );
    crate::audio::voice_verifier::assert_weight_tier(
        &per_negative_sequence_weights,
        n_ambient + n_owner + n_unrelated,
        n_confusable,
        crate::audio::voice_verifier::CONFUSABLE_UPWEIGHT,
        "confusable",
    );
    assert_eq!(per_negative_sequence_weights.len(), n_seq_total);

    info!(
        "Built {} verifier neg sequences ({} ambient@1.0× + {} owner@{}× + {} unrelated@{}× + {} confusable@{}×) \
         and {} classifier neg sequences (mahbot-932 Fix 8)",
        n_seq_total,
        n_ambient,
        n_owner,
        crate::audio::voice_verifier::OWNER_NEGATIVE_UPWEIGHT,
        n_unrelated,
        crate::audio::voice_verifier::UNRELATED_UPWEIGHT,
        n_confusable,
        crate::audio::voice_verifier::CONFUSABLE_UPWEIGHT,
        n_dense_total,
    );

    // ── Regression assertions (mahbot-932) ──
    assert!(
        classifier_neg_sequences
            .iter()
            .any(|s| s.source == Source::Ambient),
        "Phase 3 must produce ambient negative sequences (mahbot-932 Fix 7)",
    );
    assert!(
        classifier_neg_sequences
            .iter()
            .any(|s| s.source == Source::Owner),
        "Phase 3 must produce owner-negative sequences (mahbot-932 Fix 6)",
    );
    assert!(
        !classifier_neg_sequences
            .iter()
            .any(|s| s.source == Source::Synthetic),
        "Phase 3 must NOT produce synthetic negatives (mahbot-932 Fix 7)",
    );

    phase_times[P_NEG_TRAINING_DATA] = phase_end_ms!();

    // ── Phase 4: finalize_enrollment (consistency check + classifier training) ──
    phase_start!("Phase 4: finalize_enrollment");
    // After mahbot-923, pos_sequences contains dense stride-8 embeddings
    // from vad_segment_and_enroll (no streaming separate path).
    // Both positives and negatives use dense-only EmbeddingSequence.
    let training_result = super::finalize_enrollment(&pos_sequences, &classifier_neg_sequences)
        .expect("finalize_enrollment must succeed — consistency check + classifier training");

    let weights = training_result.weights.clone();
    let epochs_trained = training_result.epochs_trained;
    let best_val_loss = training_result.best_val_loss;
    let pos_scores_mean = training_result.pos_scores_mean;
    let pos_scores_min = training_result.pos_scores_min;
    let pos_scores_max = training_result.pos_scores_max;
    let neg_scores_mean = training_result.neg_scores_mean;
    let neg_scores_min = training_result.neg_scores_min;
    let neg_scores_max = training_result.neg_scores_max;

    let classifier = WakeWordClassifier::new(weights.clone());
    let first_params = weights.param_count();
    info!(
        "Conv1D classifier trained successfully: {} params, {} epochs, best val loss={best_val_loss:.4}",
        first_params, epochs_trained,
    );
    info!(
        "Classifier scores: pos mean={pos_scores_mean:.4} [{pos_scores_min:.4}, {pos_scores_max:.4}] \
         neg mean={neg_scores_mean:.4} [{neg_scores_min:.4}, {neg_scores_max:.4}]",
    );

    // ── Degenerate solution detection (mahbot-844) ──
    // Check if the classifier has all near-zero weights, indicating a
    // degenerate all-zero solution.
    let (degenerate, near_zero_frac) = {
        let all_w = weights.all_trainable_slices();
        let total_params: usize = all_w.iter().map(|s| s.len()).sum();
        let near_zero_count = all_w
            .iter()
            .flat_map(|s| s.iter())
            .filter(|v| v.abs() < 1e-6)
            .count();
        let frac = near_zero_count as f64 / total_params as f64;
        // Degenerate if >99% of weights are within ±1e-6 of zero
        if frac > 0.99 {
            warn!(
                "Classifier produced degenerate all-zero solution — training had issues. \
                 {:.1}% of weights near zero (threshold=1%). \
                 Skipping all detection phases.",
                frac * 100.0,
            );
            (true, frac)
        } else {
            info!(
                "Classifier degenerate check: {:.1}% weights near zero — OK",
                frac * 100.0
            );
            (false, frac)
        }
    };

    // ── Informational self-test (mahbot-1006 J) ──
    // Production treats the self-test as GATING: if it fails, the model is
    // rejected and enrollment fails.  The benchmark is report-only (mahbot-953),
    // so it does not abort — but a failure is surfaced as a prominent warning
    // AND recorded in the JSON so consumers know the reported detection/FA
    // numbers come from a model production would refuse to deploy.
    let self_test_result = super::run_enrollment_self_test(&pos_sequences, &classifier);
    match &self_test_result {
        Ok(()) => info!("Detection self-test: passed"),
        Err(e) => warn!(
            "Detection self-test FAILED — production would reject this model (report-only, mahbot-953): {e}"
        ),
    }
    phase_times[P_CLASSIFIER_TRAINING] = phase_end_ms!();

    // ── Phase 5: Train the VoiceVerifier (mahbot-855, mahbot-861) ─────────
    phase_start!("Phase 5: Training VoiceVerifier");

    let (verifier, verifier_metrics) = VoiceVerifier::train_with_metrics(
        &pos_sequences,
        &verifier_neg_sequences,
        Some(&per_negative_sequence_weights), // per-sequence weights matching production (mahbot-870 Fix 3)
        DEFAULT_VERIFIER_THRESHOLD,
        CONV_L2_LAMBDA, // Conv1D L2 regularization
        // mahbot-1006 K: match production's seed policy.  Production passes
        // None (entropy-based RNG) — the fixed Some(42) made the benchmark
        // deterministic but unrepresentative of the outcome distribution
        // production users actually see.  The classifier seed stays fixed
        // (Some(0) inside finalize_enrollment, identical in both paths).
        None,
    );

    if verifier.is_trained() {
        info!(
            "VoiceVerifier trained successfully with {} dense positive + {} negative sequences \
             (ambient + owner-negative + confusable + unrelated, mahbot-932).  \
             Calibrated threshold={:.4}.",
            pos_sequences.len(),
            verifier_neg_sequences.len(),
            verifier.threshold,
        );
    } else {
        warn!("VoiceVerifier is untrained (insufficient data)");
    }
    let verifier_training_threshold = verifier.threshold;
    phase_times[P_VERIFIER_TRAINING] = phase_end_ms!();

    // ── Phase 6: Set global state for streaming detection ────────────────
    phase_start!("Phase 6: Setting global state");
    super::set_classifier_weights(weights.clone());
    super::set_verifier(verifier.clone());
    phase_times[P_GLOBAL_STATE] = phase_end_ms!();

    // ── Phase 7: Streaming detection setup ────────────────────────────────
    phase_start!("Phase 7: Streaming detection setup");
    // This phase is mostly a no-op — the setup was already done in Phase 6.
    // The timing will be near-zero, which is expected.
    phase_times[P_STREAMING_SETUP] = phase_end_ms!();

    // Collect held-out positive wake variants for detection testing (mahbot-911).
    // These are the remaining variants after the PCM-augmented train/test split
    // (mahbot-932 Fix 5), NOT the variants used for classifier/verifier training.
    let pos_test_variants = test_variants;

    // ── Detection phases (skipped entirely if classifier is degenerate) ─
    //
    // Initialize output vars to empty defaults — filled in below only when
    // `!degenerate`.  The `phase_times` entries for skipped phases are set
    // to 0 directly, while `phase_start!`/`phase_end_ms!` are only called
    // for executed phases.
    let mut pos_metrics = DetectionMetrics::default();
    // Cold-start detection metrics (mahbot-1006 D) — populated in Phase 8
    // alongside the warm pass.  Stays default when degenerate.
    let mut cold_metrics = DetectionMetrics::default();
    let mut conf_metrics = DetectionMetrics::default();
    let mut unrelated_metrics = DetectionMetrics::default();
    let mut silence_metric = DetectionMetrics::default();
    let mut noise_false_accepts: Vec<String> = Vec::new();
    let mut volume_sweep_results: Vec<(&'static str, f64, Vec<f32>, Vec<f32>)> = Vec::new();
    let mut mid_utterance_results: Vec<(&str, bool)> = Vec::new();
    let mut noise_overlap_results: Vec<(String, f64, Vec<serde_json::Value>)> = Vec::new();
    // Per-tier confusable fa tracking (mahbot-871). Populated after Phase 9.
    let mut conf_fa_by_tier: [Vec<String>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let conf_fa_hard_variants: Vec<(Vec<f32>, String)>;

    // Latency stats — declared here (for scope) but populated inside the
    // detection block (or remain at defaults when degenerate).
    let (mut lat_mean, mut lat_median, mut lat_p95) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut latency_samples = 0usize;

    // Cooldown-phase detection time — hoisted from the detection block so the
    // JSON report can emit it even when the classifier is degenerate
    // (mahbot-1005 §8).
    let mut cooldown_detection_time_ms = 0.0f64;
    // Cooldown test outcomes (None = test skipped).  Emitted in the JSON
    // report so cooldown behaviour is observable (mahbot-1005 §8).
    let mut cooldown_first_detected: Option<bool> = None;
    let mut cooldown_suppressed: Option<bool> = None;
    let mut cooldown_after_recovered: Option<bool> = None;

    // Per-variant metrics for noise profiles (collected across all profiles).
    let mut noise_metrics: Vec<DetectionMetrics> = Vec::new();

    if !degenerate {
        // Create a pre-warmed adaptive threshold state shared across all
        // detection phases so the adaptive code path is exercised end-to-end
        // (mahbot-845, reviewer_3).  Without this, test_detection_samples
        // creates a fresh PipelineCtx per variant, keeping the adaptive state
        // in perpetual 5-frame bootstrap and measuring all metrics against the
        // static threshold.
        //
        // Note (mahbot-852): The adaptive threshold no longer feeds high
        // classifier scores into its statistics — only background frames
        // (below NO_MATCH_RESET_THRESHOLD) update the running statistics.
        // Positive variants therefore no longer inflate the adaptive
        // threshold, eliminating the bias described in reviewer_3's
        // methodological note.  The noise-overlap phase (14) uses a
        // separate freshly-warmed state for independent measurement.
        let mut shared_adaptive = super::AdaptiveThresholdState::warmed();

        // ── Phase 8: Detection — Positive cases ───────────────────────────
        phase_start!("Phase 8: Positive (wake word) variants");
        // Warm pass (existing behavior): consume_warmup + shared pre-warmed
        // adaptive state.  Measures the optimistic production scenario where
        // the verifier is active before the utterance (background audio has
        // already filled the ring).
        test_detection_samples(
            &pos_test_variants,
            &classifier,
            &verifier,
            &mut pos_metrics,
            |m, _| m.detected += 1,
            Some(&mut shared_adaptive),
            false, // warm pass
        );
        info!(
            "Positive detection (warm): {}/{} ({:.1}%)",
            pos_metrics.detected,
            pos_metrics.total,
            pos_metrics.detection_rate() * 100.0,
        );

        // ── Cold-start pass (mahbot-1006 D) ─────────────────────────────
        // Production has no warm-up: after ≥SEGMENT_TIMEOUT_HOPS of silence the
        // segment boundary resets the ring, and the first
        // VERIFIER_WARMUP_EMBEDDINGS embeddings of the next utterance are
        // suppressed during verifier warm-up.  A short wake word therefore has
        // only a few scorable embeddings.  The warm pass cannot measure this —
        // it pre-warms the verifier, the AGC, and the classifier window.
        //
        // The cold pass re-runs the positive variants with a fresh PipelineCtx
        // per variant, no consume_warmup, and a fresh AdaptiveThresholdState
        // that bootstraps from the first real frames (mahbot-1006 F).  Scoped
        // to positive variants only: the negative phases measure false accepts
        // (which the cold start does not materially change) and doubling every
        // phase would roughly double benchmark runtime for marginal benefit.
        // NOTE: reuses the outer `cold_metrics` (declared with pos_metrics
        // above) — a shadowing `let` here would leave the JSON report reading
        // the empty outer binding (analyst review, mahbot-1006).
        test_detection_samples(
            &pos_test_variants,
            &classifier,
            &verifier,
            &mut cold_metrics,
            |m, _| m.detected += 1,
            None, // fresh adaptive state per variant (item F)
            true, // cold start
        );
        info!(
            "Positive detection (cold): {}/{} ({:.1}%)",
            cold_metrics.detected,
            cold_metrics.total,
            cold_metrics.detection_rate() * 100.0,
        );
        phase_times[P_POSITIVE_VARIANTS] = phase_end_ms!();

        // ── Phase 9: Detection — Confusable phrases ──────────────────────
        phase_start!("Phase 9: Negative — confusable phrases");
        let confusable_variants = generate_phrase_variants_cached(
            CONFUSABLE_PHRASES,
            &available_styles,
            SeedConfig {
                base_seed: 800,
                num_variants: 1, // single seed per phrase (detection test, not training)
                seed_variant: 0,
            },
            "confusable",
            &model_version_hash,
            &cache_dir_path,
        );
        info!(
            "Generated {} confusable phrase variants",
            confusable_variants.len()
        );
        test_detection_samples(
            &confusable_variants,
            &classifier,
            &verifier,
            &mut conf_metrics,
            |m, l| m.false_accepts.push(l.to_string()),
            Some(&mut shared_adaptive),
            false, // warm pass only (negative phase)
        );
        // Split false accepts by tier for per-tier tracking (mahbot-871).
        for label in &conf_metrics.false_accepts {
            let phrase = phrase_from_label(label, "confusable");
            let tier_idx = tier_for_phrase(phrase).index();
            conf_fa_by_tier[tier_idx].push(label.clone());
        }
        // Build hard-tier confusable variants slice for mid-utterance test.
        conf_fa_hard_variants = confusable_variants
            .iter()
            .filter(|(_, label)| {
                let phrase = phrase_from_label(label, "confusable");
                CONFUSABLE_HARD.contains(&phrase)
            })
            .cloned()
            .collect();
        info!(
            "Hard-tier confusable variants: {} (for mid-utterance test)",
            conf_fa_hard_variants.len()
        );
        phase_times[P_CONFUSABLE_NEGATIVES] = phase_end_ms!();

        // ── Phase 10: Detection — Unrelated phrases ───────────────────────
        phase_start!("Phase 10: Negative — unrelated phrases");
        let unrelated_variants = generate_phrase_variants_cached(
            UNRELATED_PHRASES,
            &available_styles,
            SeedConfig {
                base_seed: 900,
                num_variants: 1, // single seed per phrase (detection test, not training)
                seed_variant: 0,
            },
            "unrelated",
            &model_version_hash,
            &cache_dir_path,
        );
        info!(
            "Generated {} unrelated phrase variants",
            unrelated_variants.len()
        );
        test_detection_samples(
            &unrelated_variants,
            &classifier,
            &verifier,
            &mut unrelated_metrics,
            |m, l| m.false_accepts.push(l.to_string()),
            Some(&mut shared_adaptive),
            false, // warm pass only (negative phase)
        );
        phase_times[P_UNRELATED_NEGATIVES] = phase_end_ms!();

        // ── Phase 11: Detection — Silence ────────────────────────────────
        phase_start!("Phase 11: Negative — silence");
        test_detection_samples(
            &[(vec![0.0f32; SILENCE_LEN], "silence".to_string())],
            &classifier,
            &verifier,
            &mut silence_metric,
            |m, l| m.false_accepts.push(l.to_string()),
            Some(&mut shared_adaptive),
            false, // warm pass only (negative phase)
        );
        phase_times[P_SILENCE_NEGATIVES] = phase_end_ms!();

        // ── Phase 12: Detection — Noise profiles ─────────────────────────
        phase_start!("Phase 12: Negative — noise profiles");
        for (label, generator) in NOISE_PROFILES {
            info!("  Testing noise profile: {label}");
            let noise = generator();
            let mut metric = DetectionMetrics::default();
            test_detection_samples(
                &[(noise, (*label).to_string())],
                &classifier,
                &verifier,
                &mut metric,
                |m, l| m.false_accepts.push(l.to_string()),
                Some(&mut shared_adaptive),
                false, // warm pass only (negative phase)
            );
            if metric.false_accepts.is_empty() {
                info!("    → no false accepts ✓");
            } else {
                info!("    → false accepts: {}", metric.false_accepts.len());
                noise_false_accepts.extend(metric.false_accepts.clone());
            }
            noise_metrics.push(metric);
        }
        phase_times[P_NOISE_PROFILES] = phase_end_ms!();

        // ── Phase 13: Cooldown verification ──────────────────────────────
        // The ~3.1s sleep is EXCLUDED from the timing measurement.
        //
        // Only Detection 1 receives warm-up prepend (mahbot-922): Detection 2
        // reuses the same ctx with cooldown active (processing hits the cooldown
        // gate and calls reset_detection_segment(), which clears the embedding
        // ring).  After cooldown expiry, Detection 3 starts with an empty ring
        // and survives warm-up naturally because the wake word utterance is well
        // over 512ms long — the first ~512ms is consumed by the built-in warm-up
        // suppression, and the remaining audio is long enough to fire detection.
        info!("─── {}. Cooldown verification ───", P_COOLDOWN + 1);
        if let Some((first_pos, _label)) = pos_test_variants.first() {
            let mut ctx = super::PipelineCtx::new();
            // Propagate the shared adaptive state accumulated across phases 8-12
            // so the adaptive code path is active during cooldown testing too.
            ctx.adaptive_threshold = shared_adaptive.clone();

            // Detection 1: should fire (consume warm-up first so the verifier
            // is active and latency measures only the wake word — mahbot-922).
            consume_warmup(&mut ctx);
            let t0 = Instant::now();
            let detected = run_streaming_detection(first_pos, &mut ctx);
            cooldown_detection_time_ms += t0.elapsed().as_secs_f64() * 1000.0;
            if !detected.detected {
                cooldown_first_detected = Some(false);
                warn!(
                    "Cooldown test: first detection should fire but didn't (detection rate is 0/{} — skipping cooldown)",
                    pos_test_variants.len()
                );
                // Skip remaining cooldown assertions since detection didn't fire.
                // The test will still exercise noise overlap and other phases.
            } else {
                cooldown_first_detected = Some(true);
                info!("Cooldown test: first detection fired ✓");

                // Detection 2: should NOT fire during cooldown
                let before_cooldown = ctx.last_wake_word_detection;
                let t1 = Instant::now();
                let silenced = run_streaming_detection(first_pos, &mut ctx);
                cooldown_detection_time_ms += t1.elapsed().as_secs_f64() * 1000.0;
                if silenced.detected {
                    cooldown_suppressed = Some(false);
                    warn!("Cooldown test: detection fired during cooldown — unexpected");
                } else {
                    cooldown_suppressed = Some(true);
                    info!("Cooldown test: cooldown prevented re-detection ✓");
                }
                if ctx.last_wake_word_detection != before_cooldown {
                    warn!("Cooldown test: last_wake_word_detection changed during cooldown");
                }

                // Wait for cooldown to expire (sleep excluded from timing)
                info!(
                    "Cooldown test: waiting {}ms for cooldown expiry...",
                    super::WAKE_WORD_COOLDOWN.as_millis()
                );
                std::thread::sleep(
                    super::WAKE_WORD_COOLDOWN + std::time::Duration::from_millis(100),
                );

                // Detection 3: should fire again after cooldown
                ctx.last_wake_word_detection = None;
                let t2 = Instant::now();
                let after_cooldown = run_streaming_detection(first_pos, &mut ctx);
                cooldown_detection_time_ms += t2.elapsed().as_secs_f64() * 1000.0;
                if !after_cooldown.detected {
                    cooldown_after_recovered = Some(false);
                    warn!("Cooldown test: detection should fire after cooldown expires but didn't");
                } else {
                    cooldown_after_recovered = Some(true);
                    info!("Cooldown test: detection fired after cooldown ✓");
                }
            } // close the if detected.detected/else block
        } else {
            warn!("Cooldown test: no positive variants available, skipping");
        }
        info!(
            "  → Phase 13 detection work completed in {:.0}ms (excl. {}ms sleep)",
            cooldown_detection_time_ms,
            super::WAKE_WORD_COOLDOWN.as_millis() + 100,
        );
        phase_times[P_COOLDOWN] = cooldown_detection_time_ms as u64;

        // ── Noise-overlapped detection (mahbot-845) ─────────────────
        // Test detection rate when wake word is mixed with background noise.
        phase_start!("Phase 14: Noise-overlapped detection");
        noise_overlap_results = run_noise_overlap_test(&pos_test_variants, &classifier, &verifier);
        phase_times[P_NOISE_OVERLAP] = phase_end_ms!();

        // ── Part 2: Latency measurement ─────────────────────────────────
        latency_samples = pos_metrics.latencies.len();
        (lat_mean, lat_median, lat_p95) = if pos_metrics.latencies.is_empty() {
            info!("Detection latency: no samples collected");
            (0.0, 0.0, 0.0)
        } else {
            let mean = pos_metrics.mean_latency().unwrap_or(0.0);
            let median = pos_metrics.median_latency().unwrap_or(0.0);
            let p95 = pos_metrics.p95_latency().unwrap_or(0.0);
            info!(
                "Detection latency: mean={mean:.1}ms median={median:.1}ms p95={p95:.1}ms (n={})",
                pos_metrics.latencies.len(),
            );
            (mean, median, p95)
        };

        // ── Part 3: Volume sweep (informational) ─────────────────────────
        info!("─── Volume sweep (informational) ───");
        volume_sweep_results = run_volume_sweep(&pos_test_variants, &classifier, &verifier);

        // ── Part 4: Mid-utterance detection (informational) ──────────────
        info!("─── Mid-utterance detection (informational) ───");
        // Always use hard-tier confusable variants as distractor (mahbot-871).
        mid_utterance_results = run_mid_utterance_test(
            &pos_test_variants,
            &conf_fa_hard_variants,
            &unrelated_variants,
        );
    } else {
        // Degenerate classifier — skip all detection phases
        phase_times[P_POSITIVE_VARIANTS] = 0;
        phase_times[P_CONFUSABLE_NEGATIVES] = 0;
        phase_times[P_UNRELATED_NEGATIVES] = 0;
        phase_times[P_SILENCE_NEGATIVES] = 0;
        phase_times[P_NOISE_PROFILES] = 0;
        phase_times[P_COOLDOWN] = 0;
        phase_times[P_NOISE_OVERLAP] = 0;
    }

    // ── Phase 15 timing ─────────────────────────────────
    phase_start!("Phase 15: Teardown");

    let tier_limit_sets = [
        tier_limits(BenchTier::Easy),
        tier_limits(BenchTier::Medium),
        tier_limits(BenchTier::Hard),
    ];

    // Compute total false accepts across all categories
    let shared_fa_count = unrelated_metrics.false_accepts.len()
        + silence_metric.false_accepts.len()
        + noise_false_accepts.len();

    // Per-tier confusable FA counts
    let conf_fa_counts = [
        conf_fa_by_tier[0].len(),
        conf_fa_by_tier[1].len(),
        conf_fa_by_tier[2].len(),
    ];

    // Per-tier total FAs (confusable + shared)
    let tier_total_fas: [usize; 3] = [
        conf_fa_counts[0] + shared_fa_count,
        conf_fa_counts[1] + shared_fa_count,
        conf_fa_counts[2] + shared_fa_count,
    ];

    // Total false accept (across all tiers and categories)
    let total_false_accepts = conf_fa_counts.iter().sum::<usize>() + shared_fa_count;

    // ── Classifier trigger metrics for negative phases (mahbot-952) ────────
    let conf_classifier = ClassifierTriggerMetrics::compute(&conf_metrics.per_variant);
    let unrel_classifier = ClassifierTriggerMetrics::compute(&unrelated_metrics.per_variant);
    let silence_classifier = ClassifierTriggerMetrics::compute(&silence_metric.per_variant);

    // Per-tier confusable classifier trigger metrics
    let mut conf_classifier_by_tier: [ClassifierTriggerMetrics; 3] = Default::default();
    for pv in &conf_metrics.per_variant {
        let phrase = phrase_from_label(&pv.variant, "confusable");
        conf_classifier_by_tier[tier_for_phrase(phrase).index()].accumulate(pv);
    }

    // Noise: per-profile + aggregate classifier trigger metrics.
    // When the classifier is degenerate, noise_metrics is empty because
    // all detection phases are skipped.  In that case produce default
    // (all-zero) metrics so downstream output paths (JSON, info!(),
    // eprintln) always have consistent data without conditional guards.
    let noise_classifier_per_profile: Vec<ClassifierTriggerMetrics> = if noise_metrics.is_empty() {
        NOISE_PROFILES
            .iter()
            .map(|_| ClassifierTriggerMetrics::default())
            .collect()
    } else {
        noise_metrics
            .iter()
            .map(|m| ClassifierTriggerMetrics::compute(&m.per_variant))
            .collect()
    };
    let noise_classifier_aggregate = noise_classifier_per_profile.iter().copied().fold(
        ClassifierTriggerMetrics::default(),
        |mut acc, m| {
            acc.total_variants += m.total_variants;
            acc.classifier_triggers += m.classifier_triggers;
            acc.warmup_suppressed += m.warmup_suppressed;
            acc.verifier_caught += m.verifier_caught;
            acc.full_pipeline_fa += m.full_pipeline_fa;
            acc
        },
    );

    // ── Verifier recall on wake-word variants (mahbot-1008 Fix 6) ──
    // The pre-fix verifier accepted 0/17 positive variants (constant 6.67e-8
    // reject-all) while the benchmark reported the 0% detection rate without a
    // verifier-recall signal.  Compute the accept rate over variants the
    // verifier actually evaluated.  Computed before the report so both the
    // human-readable report and the JSON output use the same value.
    let verifier_recall_rate = verifier_recall(&pos_metrics.per_variant);

    info!("══════════════════════════════════════════════");
    info!("      Voice Pipeline E2E Benchmark Results");
    info!("══════════════════════════════════════════════");
    info!(
        "Total benchmark time: {:.1}s",
        overall_start.elapsed().as_secs_f64()
    );
    info!(
        "Detection rate: {:.1}% ({}/{}) — target ≥{:.0}%",
        pos_metrics.detection_rate() * 100.0,
        pos_metrics.detected,
        pos_metrics.total,
        MIN_DETECTION_RATE * 100.0,
    );
    // ── Verifier recall report (mahbot-1008 Fix 6) ──
    // Report-only warning (mahbot-953 precedent): a verifier that rejects most
    // genuine wake words it evaluates is a brick wall regardless of the
    // overall detection rate.  An untrained (no-op) verifier reports N/A —
    // there is no gate to measure.
    match verifier_recall_rate {
        Some((accepted, evaluated)) => {
            let r = accepted as f64 / evaluated as f64;
            info!(
                "Verifier recall: {:.1}% ({accepted}/{evaluated}) — minimum ≥{:.0}% \
                 (over variants the verifier actually evaluated)",
                r * 100.0,
                VERIFIER_RECALL_MIN * 100.0,
            );
            if r < VERIFIER_RECALL_MIN {
                warn!(
                    "Verifier recall BELOW minimum: {:.1}% < {:.0}% (mahbot-1008 Fix 6). \
                     The verifier rejects too many genuine wake words — check the \
                     verifier_diagnostics.training block for held-out TPR.",
                    r * 100.0,
                    VERIFIER_RECALL_MIN * 100.0,
                );
            }
        }
        None => {
            info!(
                "Verifier recall: N/A (verifier untrained or no variant reached the \
                 verifier gate) — no second-stage gate was active for these variants."
            );
        }
    }
    info!(
        "Confusable false accepts: easy={} (limit ≤{}), medium={} (limit ≤{}), hard={} (limit ≤{})",
        conf_fa_counts[0],
        tier_limit_sets[0].confusable,
        conf_fa_counts[1],
        tier_limit_sets[1].confusable,
        conf_fa_counts[2],
        tier_limit_sets[2].confusable,
    );
    if !conf_metrics.false_accepts.is_empty() {
        info!(
            "  All confusable false triggers: {:?}",
            conf_metrics.false_accepts
        );
        for (i, tier_name) in [(0, "easy"), (1, "medium"), (2, "hard")] {
            if !conf_fa_by_tier[i].is_empty() {
                info!("  {tier_name}-tier triggers: {:?}", conf_fa_by_tier[i]);
            }
        }
    }
    info!(
        "Unrelated false accepts: {} / {} (limit ≤{})",
        unrelated_metrics.false_accepts.len(),
        unrelated_metrics.total,
        0, // all tiers have unrelated limit = 0
    );
    if !unrelated_metrics.false_accepts.is_empty() {
        info!("  False triggers: {:?}", unrelated_metrics.false_accepts);
    }
    info!(
        "Silence false accepts: {} / 1 (limit ≤{})",
        silence_metric.false_accepts.len(),
        0, // all tiers have silence limit = 0
    );
    info!(
        "Noise false accepts: {} / {} ({} profiles, limit ≤{}–{} across tiers)",
        noise_false_accepts.len(),
        NOISE_PROFILES.len(),
        NOISE_PROFILES.len(),
        tier_limit_sets[0].noise,
        tier_limit_sets[2].noise,
    );
    if !noise_false_accepts.is_empty() {
        info!("  False triggers: {:?}", noise_false_accepts);
    }

    // ── Classifier trigger report (mahbot-952) ──────────────────────────
    info!("──────────────────────────────────────────────");
    info!("  Classifier trigger metrics (negative phases):");

    let pr = |ct: &ClassifierTriggerMetrics| -> String {
        match ct.prevention_rate() {
            Some(r) => format!("{:.0}%", r * 100.0),
            None => "N/A".to_string(),
        }
    };
    let ct_line = |name: &str, ct: &ClassifierTriggerMetrics| -> String {
        format!(
            "    {name}: {tr}/{tot} triggers, warmup_suppressed={ws}, verifier_caught={vc}, fa={fa}, prevention={pr}",
            tr = ct.classifier_triggers,
            tot = ct.total_variants,
            ws = ct.warmup_suppressed,
            vc = ct.verifier_caught,
            fa = ct.full_pipeline_fa,
            pr = pr(ct),
        )
    };
    info!("{}", ct_line("Confusable", &conf_classifier));
    for (i, tier_name) in [(0, "easy"), (1, "medium"), (2, "hard")] {
        info!(
            "    └─ {}",
            ct_line(tier_name, &conf_classifier_by_tier[i]).trim_start()
        );
    }
    info!("{}", ct_line("Unrelated", &unrel_classifier));
    info!("{}", ct_line("Silence", &silence_classifier));
    info!(
        "{}",
        ct_line("Noise (aggregate)", &noise_classifier_aggregate)
    );
    for i in 0..NOISE_PROFILES.len() {
        let profile_ct = &noise_classifier_per_profile[i];
        info!(
            "    └─ {}",
            ct_line(NOISE_PROFILES[i].0, profile_ct).trim_start()
        );
    }

    info!("──────────────────────────────────────────────");
    for (i, name) in [(0, "easy"), (1, "medium"), (2, "hard")] {
        info!(
            "Tier {name}: confusable FA={} (limit ≤{}), total FA={} (limit ≤{})",
            conf_fa_counts[i],
            tier_limit_sets[i].confusable,
            tier_total_fas[i],
            tier_limit_sets[i].total,
        );
    }
    info!("Total false accepts across all tiers: {total_false_accepts}");
    info!("Enrollment consistency: validated by finalize_enrollment (Phase 4)");
    info!("══════════════════════════════════════════════");

    // ── Phase 15 timing ─────────────────────────────────
    phase_times[P_TEARDOWN] = phase_end_ms!();

    // ═══════════════════════════════════════════════════════════════════════
    // Post-hoc verifier threshold analysis (mahbot-997)
    // ═══════════════════════════════════════════════════════════════════════

    // Collect positive verifier scores for post-hoc optimal threshold sweep.
    let pos_verifier_scores: Vec<f32> = pos_metrics
        .per_variant
        .iter()
        .map(|pv| pv.verifier_score)
        .collect();

    let n_pos_total = pos_verifier_scores.len();
    let benchmark_detection_rate = if n_pos_total > 0 {
        pos_metrics.detected as f64 / n_pos_total as f64
    } else {
        0.0
    };

    // Collect negative verifier scores across ALL negative categories so the
    // threshold sweep accounts for false-accept impact (mahbot-1005 §7).  The
    // old sweep optimised positives only, which is blind to false accepts.
    let mut neg_verifier_scores: Vec<f32> = Vec::new();
    for pv in &conf_metrics.per_variant {
        neg_verifier_scores.push(pv.verifier_score);
    }
    for pv in &unrelated_metrics.per_variant {
        neg_verifier_scores.push(pv.verifier_score);
    }
    for pv in &silence_metric.per_variant {
        neg_verifier_scores.push(pv.verifier_score);
    }
    for metric in &noise_metrics {
        for pv in &metric.per_variant {
            neg_verifier_scores.push(pv.verifier_score);
        }
    }
    let n_neg_total = neg_verifier_scores.len();

    // ── Dual sweep (mahbot-1005 §7) ────────────────────────────────────────
    // Sweep BOTH positives and negatives, mirroring the training calibration:
    // maximise TPR - λ×FPR subject to TPR ≥ 0.90, preferring higher thresholds
    // on ties (fewer false accepts).  This replaces the positives-only sweep
    // that ignored false-accept impact entirely.
    let sweep_steps = (1.0 / CALIBRATION_SWEEP_STEP).round() as usize + 1;
    let mut benchmark_dual_threshold = verifier_training_threshold;
    let mut dual_sweep_dr = benchmark_detection_rate;
    let mut dual_sweep_far = if n_neg_total > 0 { 1.0 } else { 0.0 };
    if n_pos_total > 0 {
        let mut best_youden = f64::NEG_INFINITY;
        for step in 0..sweep_steps {
            let t = step as f32 * CALIBRATION_SWEEP_STEP;
            let tp = pos_verifier_scores.iter().filter(|&&s| s >= t).count();
            let fp = neg_verifier_scores.iter().filter(|&&s| s >= t).count();
            let tpr = tp as f64 / n_pos_total as f64;
            let fpr = if n_neg_total > 0 {
                fp as f64 / n_neg_total as f64
            } else {
                0.0
            };
            let youden = tpr - f64::from(CALIBRATION_LAMBDA) * fpr;
            let feasible = tpr >= 0.90;
            // Lexicographic maximisation over (Youden, threshold): prefer the
            // higher Youden, breaking ties toward the higher threshold (fewer
            // false accepts).  Tuple comparison avoids clippy::float_cmp on
            // the original `youden == best_youden` tie-break.
            let candidate = (youden, f64::from(t));
            let best = (best_youden, f64::from(benchmark_dual_threshold));
            if feasible && candidate > best {
                best_youden = youden;
                benchmark_dual_threshold = t;
                dual_sweep_dr = tpr;
                dual_sweep_far = fpr;
            }
        }
    }

    // Compute the suggested MIN_DETECTION_RATE constant.
    // Formula: floor(mean_rate / 0.2) × 0.2, where mean_rate is the detection
    // rate from this benchmark run.  Granularity 0.2 (20%) reflects the
    // ~17-variant test set where each miss costs ~6 percentage points.
    let computed_min_dr = if n_pos_total > 0 {
        let dr_f64 = benchmark_detection_rate;
        // Add epsilon before floor() to guard against IEEE 754 imprecision:
        // e.g., 0.80 / 0.2 = 3.999999... would floor to 3 without epsilon.
        ((dr_f64 / 0.2 + 1e-12).floor() * 0.2 * 1000.0).round() / 1000.0
    } else {
        0.0
    };

    let threshold_divergence = (benchmark_dual_threshold - verifier_training_threshold).abs();

    info!(
        "Verifier threshold analysis (mahbot-997): \
         training_threshold={verifier_training_threshold:.4}, \
         benchmark-optimal threshold (post-hoc dual sweep)={benchmark_dual_threshold:.4}, \
         divergence={threshold_divergence:.4}",
    );
    info!(
        "MIN_DETECTION_RATE hint: current_constant={MIN_DETECTION_RATE:.2}, \
         computed={computed_min_dr:.3} (= floor({:.3} / 0.2) × 0.2).  \
         Update MIN_DETECTION_RATE constant if this run is representative.",
        benchmark_detection_rate,
    );
    // ── Ratchet guard (mahbot-1008) ──
    // The mahbot-997 auto-suggestion formula floors to 0.0 on failing runs,
    // which would endorse 0% detection as passing by ratcheting the constant
    // toward zero.  A failing run must never lower the bar.
    if benchmark_detection_rate < MIN_DETECTION_RATE {
        warn!(
            "Detection rate {:.1}% is BELOW the current MIN_DETECTION_RATE constant \
             ({:.0}%) — do NOT lower MIN_DETECTION_RATE based on this failing run \
             (mahbot-1008 ratchet guard).",
            benchmark_detection_rate * 100.0,
            MIN_DETECTION_RATE * 100.0,
        );
    }

    // If training-optimal and benchmark-optimal thresholds diverge by >0.05,
    // suggest λ adjustment in Stage 1 calibration.
    if verifier.is_trained() && threshold_divergence > 0.05 {
        let suggested_lambda =
            CALIBRATION_LAMBDA * (benchmark_dual_threshold / verifier_training_threshold);
        warn!(
            "Verifier threshold divergence > 0.05: training={verifier_training_threshold:.4} vs \
             benchmark={benchmark_dual_threshold:.4} (Δ={threshold_divergence:.4}).  \
             Consider adjusting Stage 1 λ from {CALIBRATION_LAMBDA:.1} to ~{suggested_lambda:.1} \
             (λ_new = λ_old × benchmark_threshold / training_threshold).  \
             Re-run benchmark to validate.",
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // JSON metrics output
    // ═══════════════════════════════════════════════════════════════════════

    // Noise-overlap: at 10 dB SNR, flag any entry whose rate falls below 0.75
    // (mahbot-845 acceptance criteria).  Warnings only — report-only (mahbot-953).
    for (key, rate, _detail) in &noise_overlap_results {
        if key.starts_with("10dB") && *rate < 0.75 {
            warn!(
                "Noise-overlap 10dB assertion FAILED: {key} rate={:.1}% (<75%)",
                rate * 100.0,
            );
        }
    }

    // NOTE: all pass/fail gating was removed in mahbot-953.  Threshold checks
    // emit warnings above but never abort the benchmark.  Run data is reported
    // in both JSON (below) and this stderr report.  See the threshold assertion
    // conversion section after the report for details (each now warns instead of
    // asserting).

    // Build the JSON output
    let mut volume_sweep_map = serde_json::Map::new();
    for (label, rate, peak_scores, rolling_sums) in &volume_sweep_results {
        volume_sweep_map.insert(
            format!("{}_rate", label.replace('-', "_")),
            serde_json::json!(rate),
        );
        volume_sweep_map.insert(
            format!("{}_peak_scores", label.replace('-', "_")),
            serde_json::json!(peak_scores),
        );
        volume_sweep_map.insert(
            format!("{}_rolling_sums", label.replace('-', "_")),
            serde_json::json!(rolling_sums),
        );
    }

    let mut mid_utterance_map = serde_json::Map::new();
    for (scenario, detected) in &mid_utterance_results {
        mid_utterance_map.insert(
            format!("{}_detected", scenario.replace('-', "_")),
            serde_json::json!(detected),
        );
    }

    // Build per-variant negative diagnostics with FULL per-variant detail
    // (mahbot-1005 §9).  Confusable variants are always tier-qualified
    // (mahbot-871).  The flat list is reused for score distributions,
    // verifier histograms, and rejection margins below.
    let mut all_neg_pv: Vec<(&PerVariantResult, String)> = Vec::new();
    for pv in &conf_metrics.per_variant {
        let phrase = phrase_from_label(&pv.variant, "confusable");
        let variant_tier = tier_for_phrase(phrase);
        all_neg_pv.push((pv, format!("confusable_{}", variant_tier.as_str())));
    }
    for pv in &unrelated_metrics.per_variant {
        all_neg_pv.push((pv, "unrelated".to_string()));
    }
    for pv in &silence_metric.per_variant {
        all_neg_pv.push((pv, "silence".to_string()));
    }
    for metric in &noise_metrics {
        for pv in &metric.per_variant {
            all_neg_pv.push((pv, "noise".to_string()));
        }
    }
    let negative_pv: Vec<serde_json::Value> = all_neg_pv
        .iter()
        .map(|(pv, cat)| pv_to_json(pv, Some(cat)))
        .collect();

    // Build per-tier results dict
    let mut results_map = serde_json::Map::new();
    for &(i, tier_name, bench_tier) in &[
        (0, "easy", BenchTier::Easy),
        (1, "medium", BenchTier::Medium),
        (2, "hard", BenchTier::Hard),
    ] {
        let limits = tier_limits(bench_tier);
        results_map.insert(
            tier_name.to_string(),
            serde_json::json!({
                "confusable_false_accepts": conf_fa_counts[i],
                "total_false_accepts": tier_total_fas[i],
                "limits": {
                    "confusable": limits.confusable,
                    "total": limits.total,
                },
            }),
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // mahbot-1005 diagnostics: verdicts, distributions, verifier evidence,
    // train/test alignment, and reproducibility metadata
    // ═══════════════════════════════════════════════════════════════════════

    // ── Exhaustive per-variant verdicts for positives (§2) ──
    let mut verdict_counts: std::collections::BTreeMap<&'static str, usize> =
        std::collections::BTreeMap::new();
    for pv in &pos_metrics.per_variant {
        *verdict_counts
            .entry(classify_miss(pv).as_str())
            .or_insert(0) += 1;
    }

    // ── Classifier discrimination evidence (§3) ──
    // Peak-score distributions (warm-up excluded after the §1 instrumentation
    // reset) plus per-frame threshold crossing fractions.
    let pos_peak_scores: Vec<f32> = pos_metrics
        .per_variant
        .iter()
        .map(|pv| pv.peak_score)
        .collect();
    let neg_peak_scores: Vec<f32> = all_neg_pv.iter().map(|(pv, _)| pv.peak_score).collect();

    let pos_frame_crossing = {
        let total_frames: usize = pos_metrics
            .per_variant
            .iter()
            .map(|pv| pv.per_frame_scores.len())
            .sum();
        let below_threshold: usize = pos_metrics
            .per_variant
            .iter()
            .flat_map(|pv| &pv.per_frame_scores)
            .filter(|s| s[ROLLING_SUM_IDX] < MIN_CLASSIFIER_THRESHOLD)
            .count();
        (total_frames, below_threshold)
    };
    let neg_frame_crossing = {
        let total_frames: usize = all_neg_pv
            .iter()
            .map(|(pv, _)| pv.per_frame_scores.len())
            .sum();
        let above_threshold: usize = all_neg_pv
            .iter()
            .flat_map(|(pv, _)| &pv.per_frame_scores)
            .filter(|s| s[ROLLING_SUM_IDX] >= MIN_CLASSIFIER_THRESHOLD)
            .count();
        (total_frames, above_threshold)
    };

    // ── Verifier evidence (§4) ──
    // Per-category verifier peak-score histograms over [0,1] with 20 buckets.
    let verifier_hist_buckets = 20usize;
    let mut verifier_hist: std::collections::BTreeMap<String, serde_json::Value> =
        std::collections::BTreeMap::new();
    let pos_vscores: Vec<f32> = pos_metrics
        .per_variant
        .iter()
        .map(|pv| pv.verifier_score)
        .collect();
    verifier_hist.insert(
        "positive".to_string(),
        serde_json::json!({
            "histogram": histogram_buckets(&pos_vscores, verifier_hist_buckets, 0.0, 1.0),
            "deciles": deciles(&pos_vscores),
            "n": pos_vscores.len(),
        }),
    );
    for cat in [
        "confusable_easy",
        "confusable_medium",
        "confusable_hard",
        "unrelated",
        "silence",
        "noise",
    ] {
        let scores: Vec<f32> = all_neg_pv
            .iter()
            .filter(|(_, c)| c == cat)
            .map(|(pv, _)| pv.verifier_score)
            .collect();
        verifier_hist.insert(
            cat.to_string(),
            serde_json::json!({
                "histogram": histogram_buckets(&scores, verifier_hist_buckets, 0.0, 1.0),
                "deciles": deciles(&scores),
                "n": scores.len(),
            }),
        );
    }

    // Candidate lifecycle totals across all positive + negative variants.
    let candidate_lifecycle = {
        let mut created = 0usize;
        let mut confirmed = 0usize;
        let mut expired = 0usize;
        for pv in pos_metrics
            .per_variant
            .iter()
            .chain(all_neg_pv.iter().map(|(pv, _)| *pv))
        {
            created += pv.candidates_created;
            confirmed += pv.candidates_confirmed;
            expired += pv.candidates_expired;
        }
        serde_json::json!({ "created": created, "confirmed": confirmed, "expired": expired })
    };

    // Rejection margins (threshold − verifier_peak) for variants that were
    // evaluated by a trained verifier and rejected (peak below threshold).
    let rejection_margins: Vec<serde_json::Value> = pos_metrics
        .per_variant
        .iter()
        .chain(all_neg_pv.iter().map(|(pv, _)| *pv))
        .filter(|pv| {
            pv.verifier_trained
                && !pv.detected
                && pv.verifier_score < pv.verifier_threshold
                && pv.verifier_score > 0.0
        })
        .map(|pv| {
            serde_json::json!({
                "variant": pv.variant,
                "verifier_peak": pv.verifier_score,
                "threshold": pv.verifier_threshold,
                "margin": pv.verifier_threshold - pv.verifier_score,
            })
        })
        .collect();

    // ── Per-augmentation detection rates (§6) ──
    // Discloses whether the tail-of-list 2/3-1/3 split (by list order) biases
    // specific augmentation types.  Labels carry `_original/_speed_down/...`
    // suffixes from pcm_augment_enrollment_variants.
    let mut aug_counts: std::collections::BTreeMap<&'static str, (usize, usize)> =
        std::collections::BTreeMap::new();
    for pv in &pos_metrics.per_variant {
        let aug = augmentation_type(&pv.variant);
        let e = aug_counts.entry(aug).or_insert((0, 0));
        e.1 += 1;
        if pv.detected {
            e.0 += 1;
        }
    }
    let per_augmentation: serde_json::Value = serde_json::Value::Object(
        aug_counts
            .iter()
            .map(|(aug, (detected, total))| {
                (
                    (*aug).to_string(),
                    serde_json::json!({
                        "detected": detected,
                        "total": total,
                        "rate": if *total > 0 { *detected as f64 / *total as f64 } else { 0.0 },
                    }),
                )
            })
            .collect(),
    );

    // ── Train/test alignment (§6) ──
    // Embedding cosine similarity between held-out test variants and the
    // training centroid.  The centroid is the L2-normalised mean of all
    // training embeddings (pos_sequences); each test variant's embeddings are
    // recomputed through the SAME processing the training path applies — a
    // fresh CONFIG-driven AudioPreprocessor (mahbot-1006 B/L) followed by
    // process_enrollment_sample — so the comparison is apples-to-apples.
    // (The detection path itself applies AGC via process_frame; this check
    // mirrors the training AGC for an embedding-space comparison.)
    let train_centroid = embedding_centroid(&pos_sequences);
    let mut test_centroid_sims: Vec<(String, f32)> = Vec::new();
    if !degenerate && let Some(centroid) = &train_centroid {
        use crate::audio::audio_preprocessor::AudioPreprocessor;
        let chunk_size = super::FRAME_LENGTH;
        for (pcm, label) in &pos_test_variants {
            let mut pre = AudioPreprocessor::new(enrollment_preprocessor_config());
            let mut agc_audio: Vec<f32> = Vec::with_capacity(pcm.len());
            for chunk in pcm.chunks(chunk_size) {
                agc_audio.extend(pre.process(chunk.to_vec()));
            }
            if let Ok(embs) = super::process_enrollment_sample(&agc_audio)
                && !embs.is_empty()
            {
                let mut mean = vec![0.0_f32; embs[0].len()];
                for e in &embs {
                    debug_assert_eq!(mean.len(), e.len());
                    for (a, b) in mean.iter_mut().zip(e) {
                        *a += b;
                    }
                }
                for v in &mut mean {
                    *v /= embs.len() as f32;
                }
                test_centroid_sims.push((label.clone(), cosine_similarity(&mean, centroid)));
            }
        }
    }
    let test_centroid_sim_stats = {
        let sims: Vec<f32> = test_centroid_sims.iter().map(|(_, s)| *s).collect();
        serde_json::json!({
            "n": sims.len(),
            "min": deciles(&sims).map(|d| d[0]),
            "mean": if sims.is_empty() { None } else {
                Some(sims.iter().copied().sum::<f32>() / sims.len() as f32)
            },
            "max": deciles(&sims).map(|d| d[9]),
            "per_variant": test_centroid_sims.iter().map(|(l, s)| {
                serde_json::json!({ "variant": l, "cosine_similarity": s })
            }).collect::<Vec<_>>(),
        })
    };

    // ── Reproducibility metadata (§10) ──
    let warmup_source = if WARMUP_TTS_CACHE.get().is_some() {
        "tts"
    } else {
        "pink_noise_fallback"
    };
    let reproducibility = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        "seeds": {
            // Classifier seed is fixed inside finalize_enrollment (voice.rs).
            "classifier": 0,
            // mahbot-1006 K: verifier seed is now null (entropy-based RNG),
            // matching production's VoiceVerifier::train(None).  The old
            // fixed Some(42) was deterministic but unrepresentative of the
            // distribution of outcomes production users see.  Run the
            // benchmark multiple times to observe verifier variance.
            "verifier": null,
            "tts_warmup": 947,
        },
        "model_hashes": {
            "tts_model_version_hash": model_version_hash,
            "mel_model_sha256": super::MEL_MODEL_SHA256,
            "embed_model_sha256": super::EMBED_MODEL_SHA256,
        },
        "cache_state": {
            "warmup_tts_cached": WARMUP_TTS_CACHE.get().is_some(),
            "cache_dir": cache_dir_path.display().to_string(),
        },
        "warmup_source": warmup_source,
        "train_test_split": {
            "n_train_clips": train_clips.len(),
            "n_test_clips": test_clips.len(),
            "n_train": pos_sequences.len(),
            "n_test": pos_test_variants.len(),
            "split": "2/3 : 1/3 by enrollment-clip list order (tail = test); \
                      clips are split BEFORE AGC/VAD/augmentation (mahbot-1006 B/I), \
                      so no augmented variant of a training clip leaks into the test set; \
                      per-augmentation rates are structurally biased toward \
                      later list positions and are disclosed, not corrected",
            "train_clips": train_clips.iter().map(|(_, l)| l.clone()).collect::<Vec<_>>(),
            "test_clips": test_clips.iter().map(|(_, l)| l.clone()).collect::<Vec<_>>(),
            "test_variants": pos_test_variants.iter().map(|(_, l)| l.clone()).collect::<Vec<_>>(),
        },
    });

    // ── JSON sub-objects (built separately to stay under serde_json's json!
    // macro recursion limit — the main report object nests several levels) ──

    // Detection summary: warm (pos_metrics) + cold-start (cold_metrics, mahbot-1006 D).
    let detection_json = serde_json::json!({
        "rate": if pos_metrics.total > 0 {
            serde_json::Value::from(pos_metrics.detection_rate())
        } else {
            serde_json::Value::Null
        },
        "detected": pos_metrics.detected,
        "total_positive": pos_metrics.total,
        "no_tests_ran": pos_metrics.total == 0,
        "miss_verdicts": verdict_counts,
        "total_misses": pos_metrics.total - pos_metrics.detected,
        // mahbot-1006 D: cold-start pass — fresh PipelineCtx per variant,
        // no consume_warmup, fresh AdaptiveThresholdState::new() bootstrap.
        // Measures production's post-silence start where the first
        // VERIFIER_WARMUP_EMBEDDINGS embeddings are suppressed.
        "cold_rate": if cold_metrics.total > 0 {
            serde_json::Value::from(cold_metrics.detection_rate())
        } else {
            serde_json::Value::Null
        },
        "cold_detected": cold_metrics.detected,
        "cold_total_positive": cold_metrics.total,
        "cold_no_tests_ran": cold_metrics.total == 0,
    });

    // mahbot-1006 J: production gates enrollment on this self-test; the
    // benchmark is report-only (mahbot-953) so it records the outcome
    // instead.  When "passed" is false, the reported detection/FA numbers
    // come from a model production would refuse to deploy.
    let self_test_json = serde_json::json!({
        "passed": self_test_result.is_ok(),
        "error": self_test_result.as_ref().err().map(ToString::to_string),
    });

    let classifier_trigger_json = serde_json::json!({
        "confusable": ct_to_json(&conf_classifier),
        "confusable_by_tier": {
            "easy": ct_to_json(&conf_classifier_by_tier[0]),
            "medium": ct_to_json(&conf_classifier_by_tier[1]),
            "hard": ct_to_json(&conf_classifier_by_tier[2]),
        },
        "unrelated": ct_to_json(&unrel_classifier),
        "silence": ct_to_json(&silence_classifier),
        "noise": {
            "aggregate": ct_to_json(&noise_classifier_aggregate),
            "per_profile": serde_json::Value::Object(
                NOISE_PROFILES.iter().enumerate().map(|(i, (label, _))| {
                    (label.to_string(), ct_to_json(&noise_classifier_per_profile[i]))
                }).collect()
            ),
        },
    });

    let classifier_diagnostics_json = serde_json::json!({
        "pos_scores_mean": pos_scores_mean,
        "pos_scores_min": pos_scores_min,
        "pos_scores_max": pos_scores_max,
        "neg_scores_mean": neg_scores_mean,
        "neg_scores_min": neg_scores_min,
        "neg_scores_max": neg_scores_max,
        "epochs_trained": epochs_trained,
        "best_val_loss": best_val_loss,
        "early_stop_reason": training_result.early_stop_reason,
        "n_train_windows": training_result.n_train_windows,
        "n_val_windows": training_result.n_val_windows,
        "per_epoch_train_loss": training_result.per_epoch_train_loss,
        "per_epoch_val_loss": training_result.per_epoch_val_loss,
        "per_epoch_val_accuracy": training_result.per_epoch_val_accuracy,
        "pos_scores_deciles": training_result.pos_scores_deciles,
        "neg_scores_deciles": training_result.neg_scores_deciles,
        "total_params": weights.param_count(),
        "degenerate": degenerate,
        "near_zero_frac": near_zero_frac,
        "adaptive_threshold_trajectory": serde_json::Value::Object(
            pos_metrics.per_variant.iter().enumerate().map(|(i, pv)| {
                (format!("variant_{i}"), serde_json::json!({
                    "variant": pv.variant,
                    "trajectory": pv.adaptive_threshold_trajectory.clone(),
                    "ceiling_limited_frames": pv.ceiling_limited_frames,
                }))
            }).collect()
        ),
        "discrimination": {
            "pos_peak_scores": {
                "deciles": deciles(&pos_peak_scores),
                "n": pos_peak_scores.len(),
            },
            "neg_peak_scores": {
                "deciles": deciles(&neg_peak_scores),
                "n": neg_peak_scores.len(),
            },
            "pos_frames_below_min_threshold_frac": if pos_frame_crossing.0 > 0 {
                pos_frame_crossing.1 as f64 / pos_frame_crossing.0 as f64
            } else {
                0.0
            },
            "neg_frames_above_min_threshold_frac": if neg_frame_crossing.0 > 0 {
                neg_frame_crossing.1 as f64 / neg_frame_crossing.0 as f64
            } else {
                0.0
            },
            "pos_total_frames": pos_frame_crossing.0,
            "neg_total_frames": neg_frame_crossing.0,
        },
    });

    let noise_overlap_json = serde_json::Value::Object(
        noise_overlap_results
            .iter()
            .map(|(key, rate, detail)| {
                (
                    key.clone(),
                    serde_json::json!({
                        "detection_rate": rate,
                        "per_variant": detail,
                    }),
                )
            })
            .collect(),
    );

    let json = serde_json::json!({
        "benchmark": "voice_pipeline_e2e",
        "_note": "Report-only benchmark — no pass/fail gating (mahbot-953). \
                  'passed' was removed in mahbot-1005: it was hardcoded true and \
                  misleading.  Compare against the limits in 'results' instead. \
                  mahbot-1006 aligned the benchmark's training/inference \
                  processing with production (AGC→augment ordering, cold-start \
                  pass, CONFIG-driven preprocessor, preprocessed negatives, \
                  verifier seed None) — detection/FA numbers are NOT directly \
                  comparable to the mahbot-1004/948 baselines.",
        "total_false_accepts": total_false_accepts,
        "results": results_map,
        "detection": detection_json,
        "self_test": self_test_json,
        "false_accepts": {
            "confusable_easy": conf_fa_counts[0],
            "confusable_medium": conf_fa_counts[1],
            "confusable_hard": conf_fa_counts[2],
            "unrelated": unrelated_metrics.false_accepts.len(),
            "silence": silence_metric.false_accepts.len(),
            "noise": noise_false_accepts.len(),
            "total": total_false_accepts,
        },
        "classifier_trigger_metrics": classifier_trigger_json,
        "phases": {
            "phase_1_enrollment_audio_ms": phase_times[P_ENROLLMENT_AUDIO],
            "phase_2_vad_enrollment_ms": phase_times[P_VAD_ENROLLMENT],
            "phase_3_negative_training_data_ms": phase_times[P_NEG_TRAINING_DATA],
            "phase_4_classifier_training_ms": phase_times[P_CLASSIFIER_TRAINING],
            "phase_5_voice_verifier_training_ms": phase_times[P_VERIFIER_TRAINING],
            "phase_6_global_state_setup_ms": phase_times[P_GLOBAL_STATE],
            "phase_7_streaming_detection_setup_ms": phase_times[P_STREAMING_SETUP],
            "phase_8_positive_variants_ms": phase_times[P_POSITIVE_VARIANTS],
            "phase_9_confusable_negatives_ms": phase_times[P_CONFUSABLE_NEGATIVES],
            "phase_10_unrelated_negatives_ms": phase_times[P_UNRELATED_NEGATIVES],
            "phase_11_silence_negatives_ms": phase_times[P_SILENCE_NEGATIVES],
            "phase_12_noise_profiles_ms": phase_times[P_NOISE_PROFILES],
            "phase_13_cooldown_ms": phase_times[P_COOLDOWN],
            "phase_14_noise_overlap_ms": phase_times[P_NOISE_OVERLAP],
            "phase_15_teardown_ms": phase_times[P_TEARDOWN],
        },
        "per_variant_results": serde_json::Value::Array(
            pos_metrics.per_variant.iter().map(|pv| pv_to_json(pv, None)).collect()
        ),
        // mahbot-1006 D: cold-start pass per-variant detail (same schema as
        // per_variant_results, but measured without consume_warmup / fresh
        // adaptive bootstrap).  Empty when the classifier is degenerate.
        "per_variant_results_cold": serde_json::Value::Array(
            cold_metrics.per_variant.iter().map(|pv| pv_to_json(pv, None)).collect()
        ),
        "per_variant_negatives": serde_json::Value::Array(negative_pv),
        "noise_overlap": noise_overlap_json,
        "latency": {
            "mean_ms": if latency_samples > 0 {
                serde_json::Value::from(lat_mean)
            } else {
                serde_json::Value::Null
            },
            "median_ms": if latency_samples > 0 {
                serde_json::Value::from(lat_median)
            } else {
                serde_json::Value::Null
            },
            "p95_ms": if latency_samples > 0 {
                serde_json::Value::from(lat_p95)
            } else {
                serde_json::Value::Null
            },
            "samples": latency_samples,
        },
        "classifier_diagnostics": classifier_diagnostics_json,
        "verifier_diagnostics": {
            "trained": verifier.is_trained(),
            "threshold": verifier_training_threshold,
            "recall": match verifier_recall_rate {
                Some((accepted, evaluated)) => serde_json::json!({
                    "accepted": accepted,
                    "evaluated": evaluated,
                    "rate": accepted as f64 / evaluated as f64,
                    "minimum": VERIFIER_RECALL_MIN,
                    "passes": (accepted as f64 / evaluated as f64) >= VERIFIER_RECALL_MIN,
                }),
                None => serde_json::json!({
                    "accepted": 0,
                    "evaluated": 0,
                    "rate": null,
                    "minimum": VERIFIER_RECALL_MIN,
                    "passes": null,
                    "note": "verifier untrained or no variant reached the verifier gate",
                }),
            },
            "score_histograms": verifier_hist,
            "candidate_lifecycle": candidate_lifecycle,
            "rejection_margins": rejection_margins,
            "training": {
                "epochs_trained": verifier_metrics.epochs_trained,
                "per_epoch_train_loss": verifier_metrics.per_epoch_train_loss,
                "per_epoch_val_loss": verifier_metrics.per_epoch_val_loss,
                "val_pos_score_mean": verifier_metrics.val_pos_score_mean,
                "val_neg_score_mean": verifier_metrics.val_neg_score_mean,
                "youden_index": verifier_metrics.youden_index,
                "tpr_at_threshold": verifier_metrics.tpr,
                "fpr_at_threshold": verifier_metrics.fpr,
                "n_val_pos": verifier_metrics.n_val_pos,
                "n_val_neg": verifier_metrics.n_val_neg,
                "threshold_calibrated": verifier_metrics.threshold_calibrated,
                "threshold_is_fallback": verifier_metrics.threshold_is_fallback,
            },
        },
        "volume_sweep": serde_json::Value::Object(volume_sweep_map),
        "mid_utterance": serde_json::Value::Object(mid_utterance_map),
        "threshold_calibration": {
            "verifier_trained": verifier.is_trained(),
            "training_threshold": verifier_training_threshold,
            "benchmark_optimal_threshold_dual": benchmark_dual_threshold,
            "dual_sweep_detection_rate_at_optimal": dual_sweep_dr,
            "dual_sweep_false_accept_rate_at_optimal": dual_sweep_far,
            "threshold_divergence": threshold_divergence,
            "computed_min_detection_rate_suggestion": computed_min_dr,
            "current_min_detection_rate_constant": MIN_DETECTION_RATE,
            "benchmark_detection_rate": benchmark_detection_rate,
        },
        "per_augmentation_detection": per_augmentation,
        "train_test_alignment": {
            "test_vs_train_centroid_cosine": test_centroid_sim_stats,
            "note": "2/3-1/3 split is by enrollment-list order; per-augmentation \
                     rates and cosine similarities are disclosed to surface any \
                     structural distribution shift between train and test.",
        },
        "cooldown": {
            "detection_time_ms": cooldown_detection_time_ms,
            "first_detection_fired": cooldown_first_detected,
            "cooldown_suppressed_redetection": cooldown_suppressed,
            "detection_recovered_after_cooldown": cooldown_after_recovered,
        },
        "reproducibility": reproducibility,
        "config": {
            "num_enrollment_variants": NUM_ENROLLMENT_VARIANTS,
            "min_detection_rate": MIN_DETECTION_RATE,
            "verifier_recall_min": VERIFIER_RECALL_MIN,
        }
    });

    // Output delimited JSON for CI tooling
    println!("--- BENCHMARK_JSON_BEGIN ---");
    println!(
        "{}",
        serde_json::to_string_pretty(&json).expect("JSON serialization")
    );
    println!("--- BENCHMARK_JSON_END ---");

    // ── Human-readable report (stderr, mahbot-871) ────────────────────────
    let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    let dr = pos_metrics.detection_rate();
    // NOTE: checkmarks are informational only — benchmark is report-only, no
    // pass/fail gating (mahbot-953).  The marks now reflect the ACTUAL limits
    // instead of being unconditional ✓ (mahbot-1005 §7).
    let mk = |pass: bool| if pass { '✓' } else { '✗' };
    let dr_pass = if pos_metrics.total > 0 && dr >= MIN_DETECTION_RATE {
        '✓'
    } else {
        '✗'
    };
    let unrelated_ok = unrelated_metrics.false_accepts.is_empty();
    let silence_ok = silence_metric.false_accepts.is_empty();
    let noise_ok = noise_false_accepts.len() <= tier_limits(BenchTier::Hard).noise;

    // Per-tier counts for report
    let fa_easy = conf_fa_by_tier[0].len();
    let fa_medium = conf_fa_by_tier[1].len();
    let fa_hard = conf_fa_by_tier[2].len();

    eprintln!(
        "\n\
         ═══════════════════════════════════════════════════════════\n\
                 Voice Pipeline E2E Benchmark Report\n\
         ═══════════════════════════════════════════════════════════\n\
         Date/Time:      {timestamp}\n\
         Tier:           Easy/Medium/Hard (per-tier limits)\n\
         Detection rate: {dr:.1}% ({detected}/{total})  {dr_pass}\n\
         MIN_DR target:  ≥{MIN_DETECTION_RATE:.0}% (suggestion: {computed_min_dr:.0}%)\n\
         Threshold:      training={verifier_training_threshold:.4}, dual-sweep-optimal={benchmark_dual_threshold:.4}\n\
         False accepts:\n\
           Confusable:\n\
             Easy:    {fa_easy}",
        detected = pos_metrics.detected,
        total = pos_metrics.total,
    );
    if !conf_fa_by_tier[0].is_empty() {
        eprintln!("               Triggers: {:?}", conf_fa_by_tier[0]);
    }
    eprintln!("             Medium:  {fa_medium}");
    if !conf_fa_by_tier[1].is_empty() {
        eprintln!("               Triggers: {:?}", conf_fa_by_tier[1]);
    }
    eprintln!("             Hard:    {fa_hard}");
    if !conf_fa_by_tier[2].is_empty() {
        eprintln!("               Triggers: {:?}", conf_fa_by_tier[2]);
    }
    eprintln!(
        "           Unrelated:  {unrelated_count}  {mark} (limit ≤0)",
        unrelated_count = unrelated_metrics.false_accepts.len(),
        mark = mk(unrelated_ok),
    );
    eprintln!(
        "           Silence:    {silence_count}  {mark} (limit ≤0)",
        silence_count = silence_metric.false_accepts.len(),
        mark = mk(silence_ok),
    );
    eprintln!(
        "           Noise:      {noise_count}  {mark} (limit ≤{noise_limit})",
        noise_count = noise_false_accepts.len(),
        mark = mk(noise_ok),
        noise_limit = tier_limits(BenchTier::Hard).noise,
    );
    eprintln!(
        "           ───────────────────────────────────────\n\
            | Easy  total: {easy_total}  {easy_mark} (limit ≤{easy_limit}) |\n\
            | Medium total: {medium_total}  {med_mark} (limit ≤{med_limit}) |\n\
            | Hard  total: {hard_total}  {hard_mark} (limit ≤{hard_limit}) |\n\
            ───────────────────────────────────────",
        easy_total = tier_total_fas[0],
        easy_mark = mk(tier_total_fas[0] <= tier_limits(BenchTier::Easy).total),
        easy_limit = tier_limits(BenchTier::Easy).total,
        medium_total = tier_total_fas[1],
        med_mark = mk(tier_total_fas[1] <= tier_limits(BenchTier::Medium).total),
        med_limit = tier_limits(BenchTier::Medium).total,
        hard_total = tier_total_fas[2],
        hard_mark = mk(tier_total_fas[2] <= tier_limits(BenchTier::Hard).total),
        hard_limit = tier_limits(BenchTier::Hard).total,
    );

    // ── Classifier trigger summary (mahbot-952) ────────────────────────
    let ct_stderr = |ct: &ClassifierTriggerMetrics| -> String {
        match ct.prevention_rate() {
            Some(r) => format!(
                "{}/{} triggers, ws={}, vc={}, fa={}, prevention={:.0}%",
                ct.classifier_triggers,
                ct.total_variants,
                ct.warmup_suppressed,
                ct.verifier_caught,
                ct.full_pipeline_fa,
                r * 100.0,
            ),
            None => format!("{}/{} triggers", ct.classifier_triggers, ct.total_variants),
        }
    };
    eprintln!(
        "         Classifier triggers:\
         \n           Confusable: {conf}\
         \n             Easy:     {easy}\
         \n             Medium:   {med}\
         \n             Hard:     {hard}\
         \n           Unrelated:  {unrel}\
         \n           Silence:    {sil}\
         \n           Noise:      {noise}",
        conf = ct_stderr(&conf_classifier),
        easy = ct_stderr(&conf_classifier_by_tier[0]),
        med = ct_stderr(&conf_classifier_by_tier[1]),
        hard = ct_stderr(&conf_classifier_by_tier[2]),
        unrel = ct_stderr(&unrel_classifier),
        sil = ct_stderr(&silence_classifier),
        noise = ct_stderr(&noise_classifier_aggregate),
    );

    eprintln!(
        "         ═══════════════════════════════════════════════════════════\n\
                   BENCHMARK COMPLETE (report-only — no pass/fail gating)\n\
         ═══════════════════════════════════════════════════════════",
    );

    // Catastrophic regression guard: at least 1 detection must occur.
    // Without this, a pipeline that detects nothing would pass all FA assertions
    // (zero detections = zero false accepts) (mahbot-911).
    // NOTE: report-only — warns instead of asserting (mahbot-953).
    if pos_metrics.detected == 0 {
        warn!(
            "Catastrophic regression: 0/{total} wake word variants detected — pipeline is not \
             detecting anything.  Detection rate: {dr:.1}%",
            total = pos_metrics.total,
            dr = pos_metrics.detection_rate() * 100.0,
        );
    }

    // Calibrated threshold check (mahbot-997).
    // The verifier threshold is now auto-calibrated during training.  This
    // informational section displays the benchmark-optimal threshold and the
    // computed MIN_DETECTION_RATE suggestion (see dual-sweep analysis above).
    let actual_dr = pos_metrics.detection_rate();
    if n_pos_total > 0 {
        info!(
            "Verifier threshold status: trained={} auto-calibrated={:.4}, \
             benchmark detection rate={:.1}% ({}/{}), \
             benchmark-optimal (dual sweep)={benchmark_dual_threshold:.4}, \
             computed MIN_DETECTION_RATE suggestion={computed_min_dr:.3}",
            verifier.is_trained(),
            verifier_training_threshold,
            actual_dr * 100.0,
            pos_metrics.detected,
            n_pos_total,
        );
    }

    // Per-tier confusable false-accept checks (mahbot-871)
    // NOTE: report-only — warns instead of asserting (mahbot-953).
    for (i, tier_name, bench_tier) in &[
        (0, "easy", BenchTier::Easy),
        (1, "medium", BenchTier::Medium),
        (2, "hard", BenchTier::Hard),
    ] {
        let limits = tier_limits(*bench_tier);
        if conf_fa_counts[*i] > limits.confusable {
            warn!(
                "Too many confusable false accepts in tier '{tier_name}': {} — need ≤{}",
                conf_fa_counts[*i], limits.confusable,
            );
        }
        if tier_total_fas[*i] > limits.total {
            warn!(
                "Too many total false accepts in tier '{tier_name}': {} — need ≤{}",
                tier_total_fas[*i], limits.total,
            );
        }
    }

    // Per-category false accept checks (non-tiered categories)
    // NOTE: report-only — warns instead of asserting (mahbot-953).
    if !unrelated_metrics.false_accepts.is_empty() {
        warn!(
            "Too many unrelated false accepts: {} — need ≤0",
            unrelated_metrics.false_accepts.len(),
        );
    }
    if !silence_metric.false_accepts.is_empty() {
        warn!(
            "Too many silence false accepts: {} — need ≤0",
            silence_metric.false_accepts.len(),
        );
    }
    if noise_false_accepts.len() > tier_limits(BenchTier::Hard).noise {
        warn!(
            "Too many noise false accepts: {} — need ≤{} (Hard-tier noise limit)",
            noise_false_accepts.len(),
            tier_limits(BenchTier::Hard).noise,
        );
    }

    // Classifier degeneracy check — report-only (mahbot-953).
    // If the classifier is degenerate, the benchmark still completes and reports
    // so we can diagnose.
    if degenerate {
        warn!(
            "Classifier produced degenerate all-zero solution — training failed. \
             {:.1}% of weights near zero (threshold=1%). \
             Detection phases were skipped. See JSON output for details.",
            near_zero_frac * 100.0,
        );
    }

    // Noise-overlap warnings already occurred at the check site above (line ~2843).
    // No pass/fail gating — report-only (mahbot-953).

    // ── Final result ──
    // All threshold checks above emit warnings on violation — no pass/fail gating (mahbot-953).
    info!("═══ E2E Voice Pipeline benchmark complete (report-only — no pass/fail gating) ═══");

    // Stop heartbeat thread and wait for it to exit.
    heartbeat_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = heartbeat_handle.join();
}

// ── Standalone warm-up validation test (mahbot-922) ──────────────────────

/// Validate that [`generate_warmup_noise`] produces enough embeddings to consume
/// the verifier warm-up period without triggering a false detection.
///
/// This is a standalone unit test (runs as part of `cargo test --features voice-tests`).
/// It requires voice model files AND TTS model files to be present in the cache
/// (from a prior app launch or benchmark run that downloaded the models).
///
/// ## Storage root setup
/// The test sets [`CONFIG.storage_root`](crate::config::ConfigReload::set_storage_root)
/// to `~/.mahbot/` so that model directories resolve correctly.  Without this, the
/// test silently skips (mahbot-947, analyst feedback).
#[test]
// Reset assertions below compare floats with exact equality on purpose: a
// fresh DetectionInstrumentation (Default) starts at exactly 0.0, so any
// nonzero peak is warm-up contamination, not rounding error.
#[expect(clippy::float_cmp)]
fn warmup_noise_produces_embeddings() {
    // Set CONFIG storage root so model paths resolve (mahbot-947).
    if crate::config::CONFIG.try_storage_root().is_none() {
        let mahbot_dir = crate::config::default_config_dir()
            .expect("Cannot resolve home directory for ~/.mahbot");
        crate::config::CONFIG.set_storage_root(mahbot_dir.clone());
        eprintln!("CONFIG storage root set to: {}", mahbot_dir.display());
    }

    // Ensure the global voice state is initialized before calling into the
    // detection pipeline, which reads classifier/verifier state from
    // `voice_state()` (mahbot-922, reviewer feedback).
    let _ = super::init_global();

    // Initialise TTS so warm-up can use TTS audio (models may or may not
    // be cached yet — we check readiness after voice model loading below).
    let _ = crate::audio::tts::init_global();

    // Skip if voice model files aren't available (requires a prior app launch
    // or benchmark run that cached the models).
    if !super::models_ready()
        && let Err(e) = ensure_voice_models_loaded()
    {
        eprintln!("Skipping warmup_noise_produces_embeddings: {e}");
        eprintln!("Run the app or E2E benchmark first to cache voice models.");
        return;
    }

    // Also skip if TTS is not ready — the primary warm-up path is TTS, and
    // the pink-noise fallback does not produce enough embeddings to satisfy
    // the assertion below.  We check *after* voice model loading so that a
    // future refactor that loads TTS models alongside voice models is handled
    // correctly (mahbot-947, reviewer feedback).
    if let Err(e) = crate::audio::tts::ensure_ready() {
        eprintln!("Skipping warmup_noise_produces_embeddings: TTS not ready — {e}");
        eprintln!("Run the app or E2E benchmark first to cache TTS models.");
        return;
    }

    let mut ctx = super::PipelineCtx::new();
    consume_warmup(&mut ctx);

    assert!(
        ctx.embedding_ring.len() >= crate::audio::voice_verifier::VERIFIER_WARMUP_EMBEDDINGS,
        "Warm-up noise should produce at least {} embeddings to consume the \
         verifier warm-up period, but only produced {}",
        crate::audio::voice_verifier::VERIFIER_WARMUP_EMBEDDINGS,
        ctx.embedding_ring.len(),
    );

    // ── mahbot-1006 A: the audio preprocessor must be reset after warm-up ──
    // The warm-up drives the AGC to a speech-adapted gain; the test utterance
    // must start from lazy-init AGC state (matching training and production's
    // reset_detection_segment).  A fresh preprocessor has fewer than 20
    // AGC-active frames, so agc_converged() returns None.  (If the assertion
    // below fails after a future change to the AGC convergence window, the
    // preprocessor may still be fresh — check agc_active_frame_count instead.)
    assert!(
        ctx.audio_preprocessor.agc_converged().is_none(),
        "AudioPreprocessor must be reset after warm-up (mahbot-1006 A): \
         the warm-up-adapted AGC state must not carry into the test utterance. \
         agc_converged() = {:?}",
        ctx.audio_preprocessor.agc_converged(),
    );

    assert!(
        ctx.last_wake_word_detection.is_none(),
        "Warm-up noise triggered a false detection (last_wake_word_detection = {:?}) — \
         the noise signal should not resemble the wake word",
        ctx.last_wake_word_detection,
    );

    // ── mahbot-1005 §1: warm-up instrumentation must be reset ─────────────
    // Warm-up audio passes through the full scoring pipeline and would
    // otherwise record per-frame scores, peaks, and candidate lifecycle
    // counters into ctx.instrumentation — contaminating the test utterance's
    // per-variant metrics (the fake "Silence: 1/1 trigger").  consume_warmup
    // must leave a fresh DetectionInstrumentation with only the warmup_*
    // evidence preserved.
    let instr = &ctx.instrumentation;
    assert!(
        instr.per_frame_scores.is_empty(),
        "Instrumentation must be reset after warm-up: per_frame_scores has {} warm-up frames",
        instr.per_frame_scores.len(),
    );
    assert!(
        instr.per_frame_verifier_scores.is_empty(),
        "Instrumentation must be reset after warm-up: per_frame_verifier_scores has {} entries",
        instr.per_frame_verifier_scores.len(),
    );
    assert_eq!(
        instr.peak_score, 0.0,
        "Instrumentation peak_score must be reset after warm-up (was {})",
        instr.peak_score,
    );
    assert_eq!(
        instr.peak_verifier_score, 0.0,
        "Instrumentation peak_verifier_score must be reset after warm-up (was {})",
        instr.peak_verifier_score,
    );
    assert_eq!(
        instr.candidates_created, 0,
        "Instrumentation candidates_created must be reset after warm-up"
    );
    assert_eq!(
        instr.vad_speech_frames, 0,
        "Instrumentation vad_speech_frames must be reset after warm-up"
    );
    assert!(
        instr.first_trigger_frame_idx.is_none(),
        "Instrumentation first_trigger_frame_idx must be reset after warm-up"
    );
    // The warm-up "head start" remains observable via the preserved fields.
    assert!(
        instr.warmup_max_rolling_sum >= 0.0,
        "warmup_max_rolling_sum must be a valid (>= 0) captured value"
    );
}

// ── classify_miss unit tests (mahbot-1005 §2) ─────────────────────────────

/// Build a `PerVariantResult` with only the fields relevant to
/// [`classify_miss`] set (defaults elsewhere).
///
/// Test-only helper: only compiled in `cargo test` builds (the `#[test]`
/// functions in this module are the sole callers), so it is `#[cfg(test)]`
/// to avoid dead-code warnings when the file is compiled as a bench/lib
/// target.  The lint expectations cover the parameter-heavy test shim.
#[cfg(test)]
#[expect(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn verdict_test_pv(
    detected: bool,
    vad_speech_frames: usize,
    n_test_embeddings: usize,
    warmup_completed: bool,
    agc_converged: Option<bool>,
    max_rolling_sum: f32,
    crossed_effective: bool,
    verifier_score: f32,
    verifier_trained: bool,
    verifier_threshold: f32,
) -> PerVariantResult {
    // n_test_embeddings is per_frame_scores.len() (mahbot-1005 §2): keep them
    // consistent so the verdict logic sees a realistic variant.
    let per_frame_scores = if n_test_embeddings == 0 {
        Vec::new()
    } else {
        // One frame: rolling_sum above MIN_CLASSIFIER_THRESHOLD; the effective
        // threshold either blocks (5.0 > rolling_sum) or is crossed (2.0).
        vec![[0.9, 2.2, if crossed_effective { 2.0 } else { 5.0 }]]
    };
    PerVariantResult {
        variant: "test".to_string(),
        detected,
        peak_score: max_rolling_sum,
        max_rolling_sum,
        verifier_score,
        n_embeddings: n_test_embeddings + 7, // preserved warm-up ring
        n_frames_below_reset: 0,
        agc_converged,
        vad_speech_frames,
        per_frame_scores,
        warmup_completed,
        warmup_n_embeddings: 7,
        verifier_score_trajectory: vec![verifier_score],
        candidates_created: 0,
        candidates_confirmed: 0,
        candidates_expired: 0,
        first_trigger_frame_idx: if crossed_effective { Some(0) } else { None },
        warmup_max_rolling_sum: 0.0,
        warmup_peak_verifier_score: 0.0,
        n_warmup_suppressed_frames: 0,
        n_test_embeddings,
        verifier_threshold,
        verifier_trained,
        adaptive_threshold_trajectory: Vec::new(),
        ceiling_limited_frames: 0,
        latency_ms: None,
    }
}

/// The 8-way exhaustive verdict must cover every combination reachable by the
/// benchmark (mahbot-1005 §2).  Also guards the fixed mis-bucketing: a miss
/// with zero VAD frames is `vad_failure`, NOT `classifier`.
#[test]
fn classify_miss_covers_all_verdicts() {
    use MissVerdict as V;

    // 1. Detected.
    assert_eq!(
        classify_miss(&verdict_test_pv(
            true,
            10,
            5,
            true,
            Some(true),
            2.5,
            true,
            0.9,
            true,
            0.6
        )),
        V::Detected,
    );
    // 2. VAD never fired → vad_failure (previously mis-bucketed as classifier).
    assert_eq!(
        classify_miss(&verdict_test_pv(
            false, 0, 0, true, None, 1.0, false, 0.0, true, 0.6
        )),
        V::VadFailure,
    );
    // 3. Zero test embeddings → vad_failure (utterance too short for 76 mel
    //    frames after the warm-up reset).
    assert_eq!(
        classify_miss(&verdict_test_pv(
            false, 3, 0, true, None, 1.0, false, 0.0, true, 0.6
        )),
        V::VadFailure,
    );
    // 4. Warm-up incomplete + classifier triggered → warmup_suppression.
    assert_eq!(
        classify_miss(&verdict_test_pv(
            false, 10, 5, false, None, 2.5, true, 0.0, true, 0.6
        )),
        V::WarmupSuppression,
    );
    // 5. AGC explicitly non-converged → agc_failure.
    assert_eq!(
        classify_miss(&verdict_test_pv(
            false,
            10,
            5,
            true,
            Some(false),
            2.5,
            true,
            0.0,
            true,
            0.6
        )),
        V::AgcFailure,
    );
    // 6. Never reached MIN_CLASSIFIER_THRESHOLD → classifier_no_trigger.
    assert_eq!(
        classify_miss(&verdict_test_pv(
            false, 10, 5, true, None, 1.5, false, 0.0, true, 0.6
        )),
        V::ClassifierNoTrigger,
    );
    // 7. Crossed hard floor but not the effective threshold → adaptive_threshold_blocked.
    assert_eq!(
        classify_miss(&verdict_test_pv(
            false, 10, 5, true, None, 2.5, false, 0.0, true, 0.6
        )),
        V::AdaptiveThresholdBlocked,
    );
    // 8. Crossed effective threshold with an untrained (no-op) verifier →
    //    verifier_timing (nothing rejected the trigger; the candidate expired).
    assert_eq!(
        classify_miss(&verdict_test_pv(
            false, 10, 5, true, None, 2.5, true, 0.0, false, 0.6
        )),
        V::VerifierTiming,
    );
    // 9. Verifier peak crossed the threshold but detection didn't fire →
    //    verifier_timing (candidate expired or audio ended first).
    assert_eq!(
        classify_miss(&verdict_test_pv(
            false, 10, 5, true, None, 2.5, true, 0.7, true, 0.6
        )),
        V::VerifierTiming,
    );
    // 10. Verifier peak below threshold → verifier_rejected.
    assert_eq!(
        classify_miss(&verdict_test_pv(
            false, 10, 5, true, None, 2.5, true, 0.3, true, 0.6
        )),
        V::VerifierRejected,
    );
}

/// `verifier_recall` must count only variants the verifier actually evaluated
/// (trained + warm-up complete + effective threshold crossed), and report the
/// accept rate among them (mahbot-1008 Fix 6).  An untrained verifier yields
/// `None` — there is no gate to measure.
#[test]
fn verifier_recall_counts_evaluated_accepts_only() {
    // Accepted: trained, warm-up done, crossed effective, peak ≥ threshold.
    let accepted = verdict_test_pv(true, 10, 5, true, None, 2.5, true, 0.9, true, 0.6);
    // Rejected: same context but peak below threshold.
    let rejected = verdict_test_pv(false, 10, 5, true, None, 2.5, true, 0.3, true, 0.6);
    // Never reached the verifier gate (effective threshold not crossed).
    let blocked = verdict_test_pv(false, 10, 5, true, None, 2.5, false, 0.0, true, 0.6);
    // Untrained verifier — not evaluated.
    let untrained = verdict_test_pv(false, 10, 5, true, None, 2.5, true, 0.9, false, 0.6);
    // Warm-up incomplete — not evaluated.
    let warmup = verdict_test_pv(false, 10, 5, false, None, 2.5, true, 0.9, true, 0.6);

    let all = vec![
        accepted.clone(),
        rejected.clone(),
        blocked.clone(),
        untrained.clone(),
        warmup.clone(),
    ];
    let (acc, ev) = verifier_recall(&all).expect("evaluated variants exist");
    assert_eq!((acc, ev), (1, 2), "only accepted+rejected are evaluated");

    // Only accepted → 100%.
    assert_eq!(verifier_recall(&[accepted]), Some((1, 1)));
    // Only rejected → 0%.
    assert_eq!(verifier_recall(&[rejected]), Some((0, 1)));
    // Nothing evaluated → None.
    assert_eq!(verifier_recall(&[blocked, untrained, warmup]), None);
}

/// `agc_converged == None` (insufficient AGC-active frames) must NOT be
/// classified as `agc_failure` — short utterances fall through to the
/// score-based verdicts (analyst feedback, mahbot-1005).
#[test]
fn classify_miss_agc_none_is_not_agc_failure() {
    use MissVerdict as V;
    assert_eq!(
        classify_miss(&verdict_test_pv(
            false, 10, 5, true, None, 2.5, true, 0.3, true, 0.6
        )),
        V::VerifierRejected,
    );
    assert_ne!(
        classify_miss(&verdict_test_pv(
            false, 10, 5, true, None, 1.5, false, 0.0, true, 0.6
        )),
        V::AgcFailure,
    );
}

// ── Non-cached generation helpers (fallback when cache unavailable) ──────
