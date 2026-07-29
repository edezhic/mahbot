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
use crate::embedding_sequence::{EmbeddingSequence, Source, UtteranceId};
use crate::tts;
use crate::voice_verifier::{DEFAULT_VERIFIER_THRESHOLD, L2_LAMBDA, VoiceVerifier};
use crate::wake_word_classifier::WakeWordClassifier;
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
/// (4 embeddings, see [`VERIFIER_WARMUP_EMBEDDINGS`](crate::voice_verifier::VERIFIER_WARMUP_EMBEDDINGS))
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
/// NOTE (mahbot-911): This is a conservatively-low placeholder value that was
/// chosen BEFORE empirical calibration on the held-out test set. The previous
/// value (0.85) was calibrated on contaminated data where the same variants
/// were used for both training and testing. After the train/test split fix,
/// this constant MUST be recalibrated by running the benchmark 3× and updating
/// to `floor(mean_rate / 0.2) * 0.2`. The 0.2 granularity reflects the ~17-variant
/// test set (after PCM augmentation × 10 enrollment variants, split 2/3:1/3)
/// where each miss costs roughly 6 percentage points.
const MIN_DETECTION_RATE: f64 = 0.60;

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
    let pcm = match crate::tts::synthesize(
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
/// If `samples` is shorter than [`FRAME_LENGTH`](super::FRAME_LENGTH), it is
/// zero-padded to match the production pipeline's behaviour when the mic
/// delivers a partial final frame.
///
/// Extracted as a shared helper to eliminate the structural near-duplicate
/// between `feed_audio` and `run_streaming_detection` (mahbot-922, reviewer
/// feedback).
fn process_frame(samples: &[f32], ctx: &mut super::PipelineCtx) {
    let chunk_size = super::FRAME_LENGTH;
    let padded = if samples.len() < chunk_size {
        let mut p = samples.to_vec();
        p.resize(chunk_size, 0.0);
        p
    } else {
        samples.to_vec()
    };
    let processed = ctx.audio_preprocessor.process(padded);
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
    // Feed silence frames to flush any remaining voice batch (matches the
    // post-audio flush in run_streaming_detection).
    for _ in 0..3 {
        process_frame(&[], ctx);
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
    if produced < crate::voice_verifier::VERIFIER_WARMUP_EMBEDDINGS {
        warn!(
            "Warm-up produced only {} embedding(s) (need >= {}) — \
             verifier will NOT be active for this variant.  \
             This is expected when using the pink-noise fallback (TTS not available). \
             Every variant's first 4 embeddings will be consumed by warm-up suppression. \
             Detection on short utterances (<1 s) will be disadvantaged.",
            produced,
            crate::voice_verifier::VERIFIER_WARMUP_EMBEDDINGS,
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
        let speed_down = crate::tts_data_gen::speed_perturbation(pcm, TARGET_SAMPLE_RATE, 0.95);
        all.push((speed_down, format!("{label}_speed_down")));

        // 3. Speed-up (1.05×, conditional — skip if too short)
        let pre_pad_samples = pcm.len().saturating_sub(2 * super::CONTEXT_PADDING_SAMPLES);
        let pre_pad_ms = (pre_pad_samples as u64 * 1000) / u64::from(TARGET_SAMPLE_RATE);
        if pre_pad_ms >= 500 {
            let speed_up = crate::tts_data_gen::speed_perturbation(pcm, TARGET_SAMPLE_RATE, 1.05);
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
fn generate_owner_negative_sequences(
    available_styles: &[String],
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Vec<EmbeddingSequence> {
    let num_styles = available_styles.len().max(1);
    let mut sequences = Vec::new();

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
                match super::process_enrollment_sample(&pcm) {
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
fn generate_ambient_noise_sequences() -> Vec<EmbeddingSequence> {
    let mut sequences = Vec::new();
    for (seq_idx, (label, noise_fn)) in NOISE_PROFILES.iter().enumerate() {
        let raw = noise_fn();
        // Level 1: full amplitude
        match super::process_enrollment_sample(&raw) {
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
        match super::process_enrollment_sample(&attenuated) {
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
            super::ENROLLMENT_VAD_THRESHOLD,
        ));
    }
    let utterances = super::segment_utterances_by_vad(
        audio,
        &vad_decisions,
        &super::DEFAULT_VAD_SEGMENTATION_CONFIG,
    );
    (vad_decisions, utterances)
}

/// Process enrollment variants through VAD-gated utterance segmentation.
///
/// Concatenates all variants with trailing silence gaps, computes VAD decisions
/// at the enrollment threshold, segments by [`segment_utterances_by_vad`], then
/// extracts embeddings from each utterance via [`process_enrollment_sample`].
///
/// This exercises the same production path as [`handle_enrollment_audio`]:
/// audio → VAD frame-by-frame → segment by VAD → per-utterance embeddings.
///
/// The trailing silence (≥1.5s) between clips ensures that
/// [`segment_utterances_by_vad`] can complete utterance boundary detection,
/// matching the production enrollment pipeline's behavior.
///
/// Returns dense-only EmbeddingSequences (stride-8) for classifier and
/// verifier training.  The old streaming path was removed in mahbot-923.
#[allow(clippy::too_many_lines)]
fn vad_segment_and_enroll(
    enrollment_variants: &[(Vec<f32>, String)],
    augmented_variants: &[(Vec<f32>, String)],
) -> Vec<EmbeddingSequence> {
    // ── Per-variant AGC (mahbot-886) ──
    // Each variant processed through a fresh AudioPreprocessor (both AGC and
    // noise suppressor), matching the production detection path.  Shared-AGC
    // approach (concatenating variants then applying AGC) created a different
    // distribution — running_rms converged during training but detection starts
    // fresh, causing 46% miss rate on TTS variants.  Per-variant fresh NS
    // avoids the same training-inference mismatch.
    use crate::audio_preprocessor::{AudioPreprocessor, PreprocessorConfig};
    let chunk_size = super::FRAME_LENGTH;

    let all_variants: Vec<&(Vec<f32>, String)> = enrollment_variants
        .iter()
        .chain(augmented_variants.iter())
        .collect();

    let mut per_variant_agc: Vec<Vec<f32>> = Vec::with_capacity(all_variants.len());
    for (samples, _label) in &all_variants {
        let mut pre = AudioPreprocessor::new(PreprocessorConfig::default());
        let mut processed: Vec<f32> = Vec::with_capacity(samples.len());
        for chunk in samples.chunks(chunk_size) {
            let padded = if chunk.len() < chunk_size {
                let mut p: Vec<f32> = chunk.to_vec();
                p.resize(chunk_size, 0.0);
                p
            } else {
                chunk.to_vec()
            };
            processed.extend(pre.process(padded));
        }
        per_variant_agc.push(processed);
    }

    // ── Concatenate AGC-processed variants with 2.0s silence gaps ──
    // 2.0s well exceeds SILENCE_THRESHOLD_SAMPLES (1.5s) for clean boundaries.
    let silence_gap_samples = (2.0 * f64::from(super::SAMPLE_RATE)) as usize;
    let silence: Vec<f32> = vec![0.0f32; silence_gap_samples];

    let mut combined_audio: Vec<f32> = Vec::new();
    for processed in &per_variant_agc {
        if !combined_audio.is_empty() {
            combined_audio.extend_from_slice(&silence);
        }
        combined_audio.extend_from_slice(processed);
    }
    // Trailing silence for the last utterance
    combined_audio.extend_from_slice(&silence);

    info!(
        "VAD concatenation: {} total samples ({:.1}s) from {} variants (per-variant AGC)",
        combined_audio.len(),
        combined_audio.len() as f64 / f64::from(super::SAMPLE_RATE),
        all_variants.len(),
    );

    let (_vad_decisions, utterances) = compute_vad_segments(&combined_audio);
    info!(
        "VAD segmentation: {} utterances from {} concatenated variants",
        utterances.len(),
        enrollment_variants.len() + augmented_variants.len(),
    );

    // ── Process each utterance through enrollment ──
    let mut dense_sequences: Vec<EmbeddingSequence> = Vec::new();

    for (i, utterance) in utterances.iter().enumerate() {
        match super::process_enrollment_sample(utterance) {
            Ok(embeddings) if !embeddings.is_empty() => {
                info!(
                    "Utterance {i}: {} samples ({:.2}s), {} embeddings",
                    utterance.len(),
                    utterance.len() as f64 / f64::from(super::SAMPLE_RATE),
                    embeddings.len(),
                );
                dense_sequences.push(EmbeddingSequence::positive(
                    UtteranceId {
                        sequence_index: i,
                        variant_index: 0,
                    },
                    Source::Enrollment,
                    None,
                    embeddings,
                ));
            }
            Ok(_) => warn!("Utterance {i}: no embeddings extracted"),
            Err(e) => warn!("Utterance {i}: embedding extraction failed: {e}"),
        }
    }

    let expected_utterances = enrollment_variants.len() + augmented_variants.len();
    info!(
        "VAD-gated enrollment: {} dense embeddings from {} utterances (expected ~{expected_utterances})",
        dense_sequences
            .iter()
            .map(|s| s.embeddings.len())
            .sum::<usize>(),
        dense_sequences.len(),
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
            };
        }
    }

    // Feed silence frames to flush any remaining voice_batch.
    // The first silence frame after speech triggers flush_voice_batch via
    // the VAD-negative branch in handle_wake_word_detection.
    // Also process through audio_preprocessor for consistency with production.
    for _ in 0..3 {
        if ctx.last_wake_word_detection != before {
            let latency = feed_start.elapsed().as_secs_f64() * 1000.0;
            return DetectionResult {
                detected: true,
                latency_ms: Some(latency),
            };
        }
        process_frame(&[], ctx);
    }

    DetectionResult {
        detected: ctx.last_wake_word_detection != before,
        latency_ms: None,
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
    /// [`VERIFIER_WARMUP_EMBEDDINGS`](crate::voice_verifier::VERIFIER_WARMUP_EMBEDDINGS)
    /// entries after `consume_warmup`).  When `false`, the verifier will NOT be
    /// active, and the first 4 embeddings from the test utterance are consumed
    /// by warm-up suppression (mahbot-947).
    warmup_completed: bool,
    /// Number of embeddings in the ring buffer after warm-up consumption,
    /// before processing the test utterance (mahbot-947).
    warmup_n_embeddings: usize,
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
) {
    // Set classifier + verifier in global state for the streaming pipeline.
    // score_stride8_window reads these from voice_state().
    super::set_classifier_weights(classifier.weights_ref().clone());
    super::set_verifier(verifier.clone());

    for (i, (samples, label)) in variants.iter().enumerate() {
        info!(
            "  Variant {}/{}: {label} — processing",
            i + 1,
            variants.len()
        );
        metrics.total += 1;
        let mut ctx = super::PipelineCtx::new();
        // Clone the shared adaptive state into this variant's ctx so the
        // adaptive threshold is active from the first frame, simulating a
        // continuous pipeline (mahbot-845, reviewer_3).  Without this the
        // adaptive state never exits its 5-frame bootstrap because each
        // variant gets a fresh ctx, keeping all benchmark metrics measured
        // against the static threshold.
        if let Some(ref mut state) = adaptive_state {
            ctx.adaptive_threshold = state.clone();
        }
        // Consume verifier warm-up before the test utterance so the latency
        // timer measures only the wake word (mahbot-922, reviewer feedback).
        consume_warmup(&mut ctx);
        let warmup_n_embeddings = ctx.embedding_ring.len();
        let warmup_completed =
            warmup_n_embeddings >= crate::voice_verifier::VERIFIER_WARMUP_EMBEDDINGS;
        info!(
            "  Variant {}/{}: {label} — warm-up {} ({} embeddings)",
            i + 1,
            variants.len(),
            if warmup_completed {
                "completed ✓"
            } else {
                "FAILED ✗"
            },
            warmup_n_embeddings,
        );
        let result = run_streaming_detection(samples, &mut ctx);
        // Propagate the updated adaptive state for the next variant.
        if let Some(ref mut state) = adaptive_state {
            **state = ctx.adaptive_threshold.clone();
        }
        let peak = ctx.instrumentation.peak_score;
        let max_rs = max_rolling_sum(&ctx.instrumentation.per_frame_scores);
        if result.detected {
            if let Some(lat) = result.latency_ms {
                metrics.latencies.push(lat);
            }
            on_detection(metrics, label);
        }
        // Record per-variant result (mahbot-845) with verifier score (mahbot-859)
        // and per-variant instrumentation (mahbot-886).
        metrics.per_variant.push(PerVariantResult {
            variant: label.clone(),
            detected: result.detected,
            peak_score: peak,
            max_rolling_sum: max_rs,
            verifier_score: ctx.instrumentation.peak_verifier_score,
            n_embeddings: ctx.embedding_ring.len(),
            n_frames_below_reset: ctx.instrumentation.n_frames_below_reset,
            agc_converged: ctx.audio_preprocessor.agc_converged(),
            vad_speech_frames: ctx.instrumentation.vad_speech_frames,
            per_frame_scores: ctx.instrumentation.per_frame_scores.clone(),
            warmup_completed,
            warmup_n_embeddings,
        });
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
/// Reports detection rate per level.  Not gated — results are reported in
/// the JSON output for analysis.
fn run_volume_sweep(
    positive_variants: &[(Vec<f32>, String)],
    classifier: &WakeWordClassifier,
    verifier: &VoiceVerifier,
) -> Vec<(&'static str, f64)> {
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
        );
        let rate = metrics.detection_rate();
        info!("Volume sweep {label} ({gain_db}dB): {:.1}%", rate * 100.0);
        results.push((*label, rate));
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
/// variants with the noise and test detection.
fn run_noise_overlap_test(
    positive_variants: &[(Vec<f32>, String)],
    classifier: &WakeWordClassifier,
    verifier: &VoiceVerifier,
) -> Vec<(String, f64)> {
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
                    warmup_n_embeddings >= crate::voice_verifier::VERIFIER_WARMUP_EMBEDDINGS;
                let result = run_streaming_detection(&mixed_pcm, &mut ctx);

                // Persist the updated adaptive state for the next variant.
                shared_adaptive = ctx.adaptive_threshold.clone();

                let peak = ctx.instrumentation.peak_score;
                let max_rs = max_rolling_sum(&ctx.instrumentation.per_frame_scores);
                if result.detected {
                    if let Some(lat) = result.latency_ms {
                        metrics.latencies.push(lat);
                    }
                    metrics.detected += 1;
                }
                metrics.per_variant.push(PerVariantResult {
                    variant: label.clone(),
                    detected: result.detected,
                    peak_score: peak,
                    max_rolling_sum: max_rs,
                    verifier_score: ctx.instrumentation.peak_verifier_score,
                    n_embeddings: ctx.embedding_ring.len(),
                    n_frames_below_reset: ctx.instrumentation.n_frames_below_reset,
                    agc_converged: ctx.audio_preprocessor.agc_converged(),
                    vad_speech_frames: ctx.instrumentation.vad_speech_frames,
                    per_frame_scores: ctx.instrumentation.per_frame_scores.clone(),
                    warmup_completed,
                    warmup_n_embeddings,
                });
            }

            let rate = metrics.detection_rate();
            let key = format!("{snr_label}_{noise_label}");
            info!(
                "Noise overlap {snr_label} / {noise_label}: {:.1}% detection ({}/{})",
                rate * 100.0,
                metrics.detected,
                metrics.total,
            );
            results.push((key, rate));
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
    crate::tts::init_global().unwrap_or_else(|e| warn!("tts::init_global() already called: {e}"));

    // Ensure voice pipeline state is initialized
    super::init_global().unwrap_or_else(|e| warn!("voice::init_global() already called: {e}"));

    // ── Prerequisites ───────────────────────────────────────────────
    if let Err(msg) = crate::tts::ensure_ready() {
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
    assert!(
        !enrollment_variants.is_empty(),
        "Need at least one enrollment variant. TTS synthesis may have failed for all styles."
    );
    info!(
        "Generated {} enrollment variants",
        enrollment_variants.len()
    );

    // Apply PCM transforms to match production: original, speed-down, speed-up
    // (conditional), volume-down, noise (mahbot-932 Fix 5).
    let all_variants = pcm_augment_enrollment_variants(&enrollment_variants);
    info!(
        "PCM augmentation: {} -> {} total variants (original + speed-down + \
         speed-up + vol-down + noise per enrollment)",
        enrollment_variants.len(),
        all_variants.len(),
    );
    phase_times[P_ENROLLMENT_AUDIO] = phase_end_ms!();

    // ── Phase 2: VAD-gated enrollment ────────────────────────────────────
    phase_start!("Phase 2: VAD-gated enrollment");

    // ── Train/test split AFTER PCM augmentation (mahbot-932 Fix 5) ──
    // Split the PCM-augmented variants into disjoint training and test sets.
    // Production generates ~45-50 sequences (10 utterances × 5 PCM variants,
    // minus SpeedUp skip for short ones).  We use a 2/3 : 1/3 split so the
    // test set has enough samples for meaningful detection metrics.
    let n_train = (all_variants.len() * 2 / 3).max(1);
    let mut all_variants_mut = all_variants;
    let test_variants: Vec<(Vec<f32>, String)> = all_variants_mut.split_off(n_train);
    let train_variants = all_variants_mut;
    let n_test = test_variants.len();
    info!(
        "Train/test split: {} training, {} test variants (mahbot-932)",
        train_variants.len(),
        n_test
    );

    let pos_sequences = vad_segment_and_enroll(&train_variants, &[]);
    assert!(
        !pos_sequences.is_empty(),
        "VAD-gated enrollment produced no utterances from {} training variants",
        train_variants.len(),
    );
    info!(
        "VAD-gated enrollment: {} dense embeddings from {} utterances",
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
        crate::voice_verifier::OWNER_NEGATIVE_UPWEIGHT,
        n_owner,
    ));
    pw.extend(std::iter::repeat_n(
        crate::voice_verifier::UNRELATED_UPWEIGHT,
        n_unrelated,
    ));
    pw.extend(std::iter::repeat_n(
        crate::voice_verifier::CONFUSABLE_UPWEIGHT,
        n_confusable,
    ));
    per_negative_sequence_weights = pw;

    // Structural guard: verify weight tier boundaries (mahbot-880).
    crate::voice_verifier::assert_weight_tier(
        &per_negative_sequence_weights,
        0,
        n_ambient,
        1.0,
        "ambient",
    );
    crate::voice_verifier::assert_weight_tier(
        &per_negative_sequence_weights,
        n_ambient,
        n_owner,
        crate::voice_verifier::OWNER_NEGATIVE_UPWEIGHT,
        "owner-negative",
    );
    crate::voice_verifier::assert_weight_tier(
        &per_negative_sequence_weights,
        n_ambient + n_owner,
        n_unrelated,
        crate::voice_verifier::UNRELATED_UPWEIGHT,
        "unrelated",
    );
    crate::voice_verifier::assert_weight_tier(
        &per_negative_sequence_weights,
        n_ambient + n_owner + n_unrelated,
        n_confusable,
        crate::voice_verifier::CONFUSABLE_UPWEIGHT,
        "confusable",
    );
    assert_eq!(per_negative_sequence_weights.len(), n_seq_total);

    info!(
        "Built {} verifier neg sequences ({} ambient@1.0× + {} owner@{}× + {} unrelated@{}× + {} confusable@{}×) \
         and {} classifier neg sequences (mahbot-932 Fix 8)",
        n_seq_total,
        n_ambient,
        n_owner,
        crate::voice_verifier::OWNER_NEGATIVE_UPWEIGHT,
        n_unrelated,
        crate::voice_verifier::UNRELATED_UPWEIGHT,
        n_confusable,
        crate::voice_verifier::CONFUSABLE_UPWEIGHT,
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

    // -- Informational self-test --
    match super::run_enrollment_self_test(&pos_sequences, &classifier) {
        Ok(()) => info!("Detection self-test (informational): passed"),
        Err(e) => info!("Detection self-test (informational, non-gating): {e}"),
    }
    phase_times[P_CLASSIFIER_TRAINING] = phase_end_ms!();

    // ── Phase 5: Train the VoiceVerifier (mahbot-855, mahbot-861) ─────────
    phase_start!("Phase 5: Training VoiceVerifier");

    let verifier = VoiceVerifier::train(
        &pos_sequences,
        &verifier_neg_sequences,
        Some(&per_negative_sequence_weights), // per-sequence weights matching production (mahbot-870 Fix 3)
        DEFAULT_VERIFIER_THRESHOLD,
        L2_LAMBDA, // mahbot-854: 0.01
        Some(42),  // fixed seed for deterministic benchmark (mahbot-882)
    );

    if verifier.is_trained() {
        info!(
            "VoiceVerifier trained successfully with {} dense positive + {} negative sequences \
             (ambient + owner-negative + confusable + unrelated, mahbot-932)",
            pos_sequences.len(),
            verifier_neg_sequences.len()
        );
    } else {
        warn!("VoiceVerifier is untrained (insufficient data)");
    }
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
    let mut conf_metrics = DetectionMetrics::default();
    let mut unrelated_metrics = DetectionMetrics::default();
    let mut silence_metric = DetectionMetrics::default();
    let mut noise_false_accepts: Vec<String> = Vec::new();
    let mut volume_sweep_results: Vec<(&str, f64)> = Vec::new();
    let mut mid_utterance_results: Vec<(&str, bool)> = Vec::new();
    let mut noise_overlap_results: Vec<(String, f64)> = Vec::new();
    // Per-tier confusable fa tracking (mahbot-871). Populated after Phase 9.
    let mut conf_fa_by_tier: [Vec<String>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let conf_fa_hard_variants: Vec<(Vec<f32>, String)>;

    // Latency stats — declared here (for scope) but populated inside the
    // detection block (or remain at defaults when degenerate).
    let (mut lat_mean, mut lat_median, mut lat_p95) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut latency_samples = 0usize;

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
        test_detection_samples(
            &pos_test_variants,
            &classifier,
            &verifier,
            &mut pos_metrics,
            |m, _| m.detected += 1,
            Some(&mut shared_adaptive),
        );
        info!(
            "Positive detection: {}/{} ({:.1}%)",
            pos_metrics.detected,
            pos_metrics.total,
            pos_metrics.detection_rate() * 100.0,
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
        let mut cooldown_detection_time_ms = 0.0f64;
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
                warn!(
                    "Cooldown test: first detection should fire but didn't (detection rate is 0/{} — skipping cooldown)",
                    pos_test_variants.len()
                );
                // Skip remaining cooldown assertions since detection didn't fire.
                // The test will still exercise noise overlap and other phases.
            } else {
                info!("Cooldown test: first detection fired ✓");

                // Detection 2: should NOT fire during cooldown
                let before_cooldown = ctx.last_wake_word_detection;
                let t1 = Instant::now();
                let silenced = run_streaming_detection(first_pos, &mut ctx);
                cooldown_detection_time_ms += t1.elapsed().as_secs_f64() * 1000.0;
                if silenced.detected {
                    warn!("Cooldown test: detection fired during cooldown — unexpected");
                } else {
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
                    warn!("Cooldown test: detection should fire after cooldown expires but didn't");
                } else {
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
    // JSON metrics output
    // ═══════════════════════════════════════════════════════════════════════

    // Noise-overlap: at 10 dB SNR, detection rate ≥ 75% for any noise type
    // (mahbot-845 acceptance criteria).  Iterate over noise_overlap_results
    // and flag any 10dB entry whose rate falls below 0.75.  This is a
    // standalone let (not inside the passed block) so it's accessible for
    // the assert!() call in the Teardown section below.
    let mut noise_overlap_10db_ok = true;
    for (key, rate) in &noise_overlap_results {
        if key.starts_with("10dB") && *rate < 0.75 {
            noise_overlap_10db_ok = false;
            warn!(
                "Noise-overlap 10dB assertion FAILED: {key} rate={:.1}% (<75%)",
                rate * 100.0,
            );
        }
    }

    // Per-tier pass/fail (mahbot-871)
    let easy_pass = conf_fa_counts[0] <= tier_limit_sets[0].confusable
        && tier_total_fas[0] <= tier_limit_sets[0].total;
    let medium_pass = conf_fa_counts[1] <= tier_limit_sets[1].confusable
        && tier_total_fas[1] <= tier_limit_sets[1].total;
    let hard_pass = conf_fa_counts[2] <= tier_limit_sets[2].confusable
        && tier_total_fas[2] <= tier_limit_sets[2].total;

    let passed = {
        // Detection rate assertion
        let dr_ok = pos_metrics.detection_rate() >= MIN_DETECTION_RATE;

        // Per-category false accept assertions (non-tiered categories)
        let unrel_fa_ok = unrelated_metrics.false_accepts.is_empty();
        let silence_fa_ok = silence_metric.false_accepts.is_empty();
        let noise_fa_ok = noise_false_accepts.len() <= tier_limit_sets[2].noise;

        dr_ok
            && easy_pass
            && medium_pass
            && hard_pass
            && unrel_fa_ok
            && silence_fa_ok
            && noise_fa_ok
            && noise_overlap_10db_ok
    };

    // Build the JSON output
    let mut volume_sweep_map = serde_json::Map::new();
    for (label, rate) in &volume_sweep_results {
        volume_sweep_map.insert(
            format!("{}_rate", label.replace('-', "_")),
            serde_json::json!(rate),
        );
    }

    let mut mid_utterance_map = serde_json::Map::new();
    for (scenario, detected) in &mid_utterance_results {
        mid_utterance_map.insert(
            format!("{}_detected", scenario.replace('-', "_")),
            serde_json::json!(detected),
        );
    }

    // Build per-variant negative diagnostics with verifier scores (mahbot-859).
    // Confusable variants are always tier-qualified (mahbot-871).
    let mut negative_pv: Vec<serde_json::Value> = Vec::new();
    for pv in &conf_metrics.per_variant {
        let phrase = phrase_from_label(&pv.variant, "confusable");
        let variant_tier = tier_for_phrase(phrase);
        let category = format!("confusable_{}", variant_tier.as_str());
        negative_pv.push(serde_json::json!({
            "variant": pv.variant,
            "detected": pv.detected,
            "peak_score": pv.peak_score,
            "max_rolling_sum": pv.max_rolling_sum,
            "verifier_score": pv.verifier_score,
            "category": category,
            "warmup_completed": pv.warmup_completed,
            "warmup_n_embeddings": pv.warmup_n_embeddings,
        }));
    }
    for pv in &unrelated_metrics.per_variant {
        negative_pv.push(serde_json::json!({
            "variant": pv.variant,
            "detected": pv.detected,
            "peak_score": pv.peak_score,
            "max_rolling_sum": pv.max_rolling_sum,
            "verifier_score": pv.verifier_score,
            "category": "unrelated",
            "warmup_completed": pv.warmup_completed,
            "warmup_n_embeddings": pv.warmup_n_embeddings,
        }));
    }
    for pv in &silence_metric.per_variant {
        negative_pv.push(serde_json::json!({
            "variant": pv.variant,
            "detected": pv.detected,
            "peak_score": pv.peak_score,
            "max_rolling_sum": pv.max_rolling_sum,
            "verifier_score": pv.verifier_score,
            "category": "silence",
            "warmup_completed": pv.warmup_completed,
            "warmup_n_embeddings": pv.warmup_n_embeddings,
        }));
    }
    for metric in &noise_metrics {
        for pv in &metric.per_variant {
            negative_pv.push(serde_json::json!({
                "variant": pv.variant,
                "detected": pv.detected,
                "peak_score": pv.peak_score,
                "max_rolling_sum": pv.max_rolling_sum,
                "verifier_score": pv.verifier_score,
                "category": "noise",
                "warmup_completed": pv.warmup_completed,
                "warmup_n_embeddings": pv.warmup_n_embeddings,
            }));
        }
    }

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

    let json = serde_json::json!({
        "benchmark": "voice_pipeline_e2e",
        "passed": passed,
        "total_false_accepts": total_false_accepts,
        "results": results_map,
        "detection": {
            "rate": pos_metrics.detection_rate(),
            "detected": pos_metrics.detected,
            "total_positive": pos_metrics.total,
            "miss_classification": {
                "classifier": pos_metrics.per_variant.iter().filter(|pv| {
                    !pv.detected
                        && !pv.per_frame_scores.iter().any(|s| s[ROLLING_SUM_IDX] >= MIN_CLASSIFIER_THRESHOLD)
                }).count(),
                "adaptive_threshold": pos_metrics.per_variant.iter().filter(|pv| {
                    !pv.detected
                        && pv.per_frame_scores.iter().any(|s| s[ROLLING_SUM_IDX] >= MIN_CLASSIFIER_THRESHOLD)
                        && !pv.per_frame_scores.iter().any(|s| s[ROLLING_SUM_IDX] >= s[THRESHOLD_IDX])
                }).count(),
                "verifier": pos_metrics.per_variant.iter().filter(|pv| {
                    !pv.detected
                        && pv.per_frame_scores.iter().any(|s| s[ROLLING_SUM_IDX] >= MIN_CLASSIFIER_THRESHOLD)
                        && pv.per_frame_scores.iter().any(|s| s[ROLLING_SUM_IDX] >= s[THRESHOLD_IDX])
                }).count(),
                "total_misses": pos_metrics.total - pos_metrics.detected,
            },
        },
        "false_accepts": {
            "confusable_easy": conf_fa_counts[0],
            "confusable_medium": conf_fa_counts[1],
            "confusable_hard": conf_fa_counts[2],
            "unrelated": unrelated_metrics.false_accepts.len(),
            "silence": silence_metric.false_accepts.len(),
            "noise": noise_false_accepts.len(),
            "total": total_false_accepts,
        },
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
            pos_metrics.per_variant.iter().map(|pv| {
                let miss_reason = if pv.detected {
                    None
                } else if !pv.per_frame_scores.iter().any(|s| s[ROLLING_SUM_IDX] >= MIN_CLASSIFIER_THRESHOLD) {
                    Some("classifier")
                } else if pv.per_frame_scores.iter().any(|s| s[ROLLING_SUM_IDX] >= s[THRESHOLD_IDX]) {
                    Some("verifier")
                } else {
                    Some("adaptive_threshold")
                };
                serde_json::json!({
                    "variant": pv.variant,
                    "detected": pv.detected,
                    "peak_score": pv.peak_score,
                    "max_rolling_sum": pv.max_rolling_sum,
                    "verifier_score": pv.verifier_score,
                    "miss_reason": miss_reason,
                    "n_embeddings": pv.n_embeddings,
                    "n_frames_below_reset": pv.n_frames_below_reset,
                    "agc_converged": pv.agc_converged,
                    "vad_speech_frames": pv.vad_speech_frames,
                    "per_frame_scores": pv.per_frame_scores,
                    "warmup_completed": pv.warmup_completed,
                    "warmup_n_embeddings": pv.warmup_n_embeddings,
                })
            }).collect()
        ),
        "per_variant_negatives": serde_json::Value::Array(negative_pv),
        "noise_overlap": serde_json::Value::Object(
            noise_overlap_results.iter().map(|(key, rate)| {
                (key.clone(), serde_json::json!(rate))
            }).collect()
        ),
        "latency": {
            "mean_ms": lat_mean,
            "median_ms": lat_median,
            "p95_ms": lat_p95,
            "samples": latency_samples,
        },
        "classifier_diagnostics": {
            "pos_scores_mean": pos_scores_mean,
            "pos_scores_min": pos_scores_min,
            "pos_scores_max": pos_scores_max,
            "neg_scores_mean": neg_scores_mean,
            "neg_scores_min": neg_scores_min,
            "neg_scores_max": neg_scores_max,
            "epochs_trained": epochs_trained,
            "best_val_loss": best_val_loss,
            "total_params": weights.param_count(),
            "degenerate": degenerate,
        },
        "volume_sweep": serde_json::Value::Object(volume_sweep_map),
        "mid_utterance": serde_json::Value::Object(mid_utterance_map),
        "config": {
            "num_enrollment_variants": NUM_ENROLLMENT_VARIANTS,
            "min_detection_rate": MIN_DETECTION_RATE,
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
    let dr_ok = dr >= MIN_DETECTION_RATE;
    let dr_pass = if dr_ok { '✓' } else { '✗' };
    let overall_pass = if passed { '✓' } else { '✗' };

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
         Tier:           Easy/Medium/Hard (per-tier assertions)\n\
         Detection rate: {dr:.1}% ({detected}/{total})  {dr_pass} (target ≥{MIN_DETECTION_RATE:.0}%)\n\
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
    let unrelated_ok = unrelated_metrics.false_accepts.is_empty();
    let silence_ok = silence_metric.false_accepts.is_empty();
    let noise_ok = noise_false_accepts.len() <= tier_limits(BenchTier::Hard).noise;
    let easy_ok = conf_fa_counts[0] <= tier_limits(BenchTier::Easy).confusable;
    let medium_ok = conf_fa_counts[1] <= tier_limits(BenchTier::Medium).confusable;
    let hard_ok = conf_fa_counts[2] <= tier_limits(BenchTier::Hard).confusable;
    eprintln!(
        "           Unrelated:  {unrelated_count}  {unrel_pass_char} (limit ≤0)",
        unrelated_count = unrelated_metrics.false_accepts.len(),
        unrel_pass_char = if unrelated_ok { '✓' } else { '✗' },
    );
    eprintln!(
        "           Silence:    {silence_count}  {sil_pass_char} (limit ≤0)",
        silence_count = silence_metric.false_accepts.len(),
        sil_pass_char = if silence_ok { '✓' } else { '✗' },
    );
    eprintln!(
        "           Noise:      {noise_count}  {noise_pass_char} (limit ≤{noise_limit})",
        noise_count = noise_false_accepts.len(),
        noise_limit = tier_limits(BenchTier::Hard).noise,
        noise_pass_char = if noise_ok { '✓' } else { '✗' },
    );
    eprintln!(
        "           ───────────────────────────────────────\n\
         \x20          | Easy  total: {easy_total}  {easy_pass_char} (limit ≤{easy_limit}) |\n\
         \x20          | Medium total: {medium_total}  {med_pass_char} (limit ≤{med_limit}) |\n\
         \x20          | Hard  total: {hard_total}  {hard_pass_char} (limit ≤{hard_limit}) |\n\
         \x20          ───────────────────────────────────────",
        easy_total = tier_total_fas[0],
        easy_limit = tier_limits(BenchTier::Easy).total,
        easy_pass_char = if easy_ok { '✓' } else { '✗' },
        medium_total = tier_total_fas[1],
        med_limit = tier_limits(BenchTier::Medium).total,
        med_pass_char = if medium_ok { '✓' } else { '✗' },
        hard_total = tier_total_fas[2],
        hard_limit = tier_limits(BenchTier::Hard).total,
        hard_pass_char = if hard_ok { '✓' } else { '✗' },
    );
    eprintln!(
        "         ═══════════════════════════════════════════════════════════\n\
                   RESULT: {overall_pass} {}\n\
         ═══════════════════════════════════════════════════════════",
        if passed { "PASS" } else { "FAIL" },
    );

    // ── Detection rate checks ──────────────────────────────────────────────
    // Catastrophic regression guard: at least 1 detection must occur.
    // Without this, a pipeline that detects nothing would pass all FA assertions
    // (zero detections = zero false accepts) and exit 0 (mahbot-911).
    assert!(
        pos_metrics.detected > 0,
        "Catastrophic regression: 0/{total} wake word variants detected — pipeline is not \
         detecting anything.  Detection rate: {dr:.1}%",
        total = pos_metrics.total,
        dr = pos_metrics.detection_rate() * 100.0,
    );

    // Calibrated threshold check — non-fatal pending recalibration (mahbot-911).
    // See MIN_DETECTION_RATE docstring for procedure.
    let actual_dr = pos_metrics.detection_rate();
    if actual_dr < MIN_DETECTION_RATE {
        warn!(
            "Detection rate {:.1}% below target ≥{:.0}% — UNTRUSTED THRESHOLD (mahbot-911) \
             — recalibrate MIN_DETECTION_RATE after 3 baseline benchmark runs",
            actual_dr * 100.0,
            MIN_DETECTION_RATE * 100.0,
        );
    }

    // Per-tier confusable false-accept assertions (mahbot-871)
    for (i, tier_name, bench_tier) in &[
        (0, "easy", BenchTier::Easy),
        (1, "medium", BenchTier::Medium),
        (2, "hard", BenchTier::Hard),
    ] {
        let limits = tier_limits(*bench_tier);
        assert!(
            conf_fa_counts[*i] <= limits.confusable,
            "Too many confusable false accepts in tier '{tier_name}': {} — need ≤{}",
            conf_fa_counts[*i],
            limits.confusable,
        );
        assert!(
            tier_total_fas[*i] <= limits.total,
            "Too many total false accepts in tier '{tier_name}': {} — need ≤{}",
            tier_total_fas[*i],
            limits.total,
        );
    }

    // Per-category false accept limits (non-tiered categories)
    assert!(
        unrelated_metrics.false_accepts.is_empty(),
        "Too many unrelated false accepts: {} — need ≤0",
        unrelated_metrics.false_accepts.len(),
    );
    assert!(
        silence_metric.false_accepts.is_empty(),
        "Too many silence false accepts: {} — need ≤0",
        silence_metric.false_accepts.len(),
    );
    assert!(
        noise_false_accepts.len() <= tier_limits(BenchTier::Hard).noise,
        "Too many noise false accepts: {} — need ≤{} (Hard-tier noise limit)",
        noise_false_accepts.len(),
        tier_limits(BenchTier::Hard).noise,
    );

    assert!(
        !degenerate,
        "Classifier produced degenerate all-zero solution — training failed. \
         {:.1}% of weights near zero (threshold=1%). \
         Detection phases were skipped. See JSON output for details.",
        near_zero_frac * 100.0,
    );

    // Noise-overlap: at 10 dB SNR, detection rate ≥ 75% for any noise type
    // (mahbot-845 acceptance criteria).
    assert!(
        noise_overlap_10db_ok,
        "Noise-overlap 10 dB SNR detection rate below 75% for at least one noise type — \
         see noise_overlap results in JSON output above for details.",
    );

    // ── Final result ──
    // The `passed` variable reflects all checks including the (non-fatal)
    // detection rate threshold — see detection rate check above (mahbot-911).
    if passed {
        info!("═══ E2E Voice Pipeline Benchmark PASSED ═══");
    } else {
        info!("═══ E2E Voice Pipeline benchmark completed — see report above for failures ═══");
    }

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
    let _ = crate::tts::init_global();

    // Skip if voice model files aren't available (requires a prior app launch
    // or benchmark run that cached the models).
    if !super::models_ready() {
        if let Err(e) = ensure_voice_models_loaded() {
            eprintln!("Skipping warmup_noise_produces_embeddings: {e}");
            eprintln!("Run the app or E2E benchmark first to cache voice models.");
            return;
        }
    }

    // Also skip if TTS is not ready — the primary warm-up path is TTS, and
    // the pink-noise fallback does not produce enough embeddings to satisfy
    // the assertion below.  We check *after* voice model loading so that a
    // future refactor that loads TTS models alongside voice models is handled
    // correctly (mahbot-947, reviewer feedback).
    if let Err(e) = crate::tts::ensure_ready() {
        eprintln!("Skipping warmup_noise_produces_embeddings: TTS not ready — {e}");
        eprintln!("Run the app or E2E benchmark first to cache TTS models.");
        return;
    }

    let mut ctx = super::PipelineCtx::new();
    consume_warmup(&mut ctx);

    assert!(
        ctx.embedding_ring.len() >= crate::voice_verifier::VERIFIER_WARMUP_EMBEDDINGS,
        "Warm-up noise should produce at least {} embeddings to consume the \
         verifier warm-up period, but only produced {}",
        crate::voice_verifier::VERIFIER_WARMUP_EMBEDDINGS,
        ctx.embedding_ring.len(),
    );

    assert!(
        ctx.last_wake_word_detection.is_none(),
        "Warm-up noise triggered a false detection (last_wake_word_detection = {:?}) — \
         the noise signal should not resemble the wake word",
        ctx.last_wake_word_detection,
    );
}

// ── Non-cached generation helpers (fallback when cache unavailable) ──────
