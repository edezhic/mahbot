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
/// 1. Run the benchmark 3× to observe verifier variance — the verifier is a
///    [`VERIFIER_ENSEMBLE_SEEDS`](crate::audio::voice_verifier::VERIFIER_ENSEMBLE_SEEDS)-member
///    multi-seed ensemble since mahbot-1025 (entropy base seed, member mean
///    scoring), matching production.
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

/// Generate additional wake-word TTS clips for the VERIFIER's positive
/// training pool (mahbot-1025).
///
/// The verifier's positive pool is the root of its seed-driven calibration
/// variance (~30–55 windows from the 6 train clips).  This helper synthesizes
/// extra wake-word clips for the SAME train voices (the first 6 styles —
/// F1..M1 in the 10-style F1-F5/M1-M5 ordering used by
/// [`generate_enrollment_variants_cached`]) at additional seeds, so the
/// verifier trains on a larger, more diverse positive pool without changing
/// the classifier's frozen enrollment positives (`pos_sequences`).
///
/// Each extra clip is also emitted as a 0.90× speed-perturbed rendition
/// (`_fast`): the enrolled speaker's speed_down (0.95×) test variant is the
/// hardest for per-member scoring (the measured miss-fraction tail comes from
/// members scoring it 0.83–0.86), so the verifier positive pool carries
/// explicitly faster speech to push the per-member fast-speech scores up.
/// A fresh-run trial with more TTS seeds (10/voice) regressed F1_noise below
/// the floor (0.83–0.87), so the pool is NOT expanded by additional seed
/// range; the speed augmentation is the targeted lever.
///
/// Returns `(samples, label)` tuples with distinct labels
/// (`{style}_verifier_extra{seed}` / `{style}_verifier_extra{seed}_fast`) so
/// provenance is clear in the report.
fn generate_extra_verifier_positives_cached(
    available_styles: &[String],
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Vec<(Vec<f32>, String)> {
    // 6 extra seeds per train voice → 36 extra clips + 36 fast renditions →
    // ~72 clips → ~360 extra positive sequences after vad_segment_and_enroll
    // (each clip yields ~5 variants).  The base pool is only ~30–55 windows
    // from the 6 train clips; the mahbot-1025 fresh-run protocol showed 2
    // extra seeds/voice left the per-member speed_down miss fraction at ~0.3
    // (gate ≤0.15) and 6 seeds at ~0.03 (mostly 0.000 with a rare 0.200 tail),
    // so the pool is expanded further to push the per-member miss probability
    // toward zero — the ticket identifies the tiny positive pool as the root
    // of the seed-driven calibration variance.
    const EXTRA_SEEDS_PER_VOICE: usize = 6;
    // Train voices = the first 6 styles (F1..M1) — mirrors the 2/3:1/3 split
    // by list order of the 10 enrollment variants.  If fewer styles exist,
    // fall back to the default style.
    let train_styles: Vec<&String> = available_styles.iter().take(6).collect();
    if train_styles.is_empty() {
        return Vec::new();
    }
    let mut variants = Vec::new();
    for (voice_idx, style) in train_styles.iter().enumerate() {
        for s in 0..EXTRA_SEEDS_PER_VOICE {
            // Distinct seed range (200+) so these never collide with the
            // base enrollment seeds (100-109) or the test variants.
            let seed = 200 + (voice_idx * EXTRA_SEEDS_PER_VOICE + s) as u64;
            if let Some(pcm) = synthesize_wake_word_variant_cached(
                WAKE_WORD,
                style,
                seed,
                TARGET_SAMPLE_RATE,
                model_hash,
                cache_dir,
            ) {
                variants.push((pcm.clone(), format!("{style}_verifier_extra{seed}")));
                // Faster rendition of the SAME clip (no extra TTS synthesis —
                // pure resampling).  The verifier then trains on explicitly
                // faster wake-word speech, which is where the per-member
                // speed_down misses live (mahbot-1025).
                let fast =
                    crate::audio::tts_data_gen::speed_perturbation(&pcm, TARGET_SAMPLE_RATE, 0.90);
                variants.push((fast, format!("{style}_verifier_extra{seed}_fast")));
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
    // ── Clear the per-segment deferred-burst latch (mahbot-1023) ──
    // Warm-up audio passes through the FULL scoring pipeline and therefore
    // triggers the deferred burst sweep on the warm-up noise.  Without
    // clearing the latch here, the test utterance's own burst sweep would be
    // suppressed (the warm-up consumption is a fresh segment boundary for the
    // benchmark's purposes).
    ctx.burst_sweep_done = false;

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
    // mahbot-1012: record the preserved warm-up ring length so the first test
    // windows (whose Conv1D window spans warm-up entries + test entries) can
    // be classified as `WindowGeometry::WarmMixed`.  Set AFTER the reset
    // above — the struct-update reset would otherwise zero it.  The embedding
    // ring itself is preserved through the reset (it is not cleared), so this
    // equals `warmup_n_embeddings` as reported to the caller.
    ctx.instrumentation.test_start_ring_len = ctx.embedding_ring.len();
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

/// Generate cross-speaker TTS wake-word negatives for verifier training
/// (mahbot-1025).
///
/// The ticket's user-approved reclassification: the noise_overlap M2–M5
/// cross-speaker cells move from held-out generalization probes to
/// **in-distribution regression canaries** — the wake word spoken by a
/// NON-enrolled TTS voice must be REJECTED by the verifier (single-speaker
/// semantics; cross-speaker generalization — Option B — remains shelved).
///
/// This helper produces the verifier's in-distribution negative sequences
/// from the held-out test clips (M2–M5): each clip's wake-word audio is
/// embedded as a [`Source::CrossSpeaker`] negative in five conditions —
/// clean, 10 dB white, 20 dB white, 10 dB pink, 20 dB pink (the ticket's
/// "clean and 10/20 dB white/pink noise-conditioned variants").  Positives
/// stay unchanged; the classifier negative set is untouched (verifier-only).
///
/// ## Preprocessing alignment
///
/// Mirrors the owner-negative path (mahbot-1009): each clip goes through a
/// fresh [`AudioPreprocessor`] (AGC + NS) in [`FRAME_LENGTH`](super::FRAME_LENGTH)
/// chunks, then the noise-conditioned variant is embedded via
/// [`super::process_enrollment_sample`] (dense stride-8 embeddings).  The
/// shared-AudioPreprocessor/`vad_gate_streaming_mel` variant used by
/// `generate_owner_negative_sequences` is deliberately NOT used here — the
/// cross-speaker negatives are wake-word AUDIO (not general speech), so the
/// enrollment-embedding path (AGC → process_enrollment_sample) matches the
/// positive-wake-word distribution the verifier must separate from.
fn generate_cross_speaker_negative_sequences(
    test_clips: &[(Vec<f32>, String)],
) -> Vec<EmbeddingSequence> {
    use crate::audio::audio_preprocessor::AudioPreprocessor;
    use crate::audio::tts_data_gen::{NoiseType, add_noise};
    let config = enrollment_preprocessor_config();
    let chunk_size = super::FRAME_LENGTH;
    let mut sequences = Vec::new();
    for (clip_idx, (pcm, label)) in test_clips.iter().enumerate() {
        // Fresh per-clip AGC (mirrors the enrollment path).
        let mut pre = AudioPreprocessor::new(config);
        let mut agc_audio: Vec<f32> = Vec::with_capacity(pcm.len());
        for chunk in pcm.chunks(chunk_size) {
            agc_audio.extend(pre.process(chunk.to_vec()));
        }
        // Conditions: clean + 10/20 dB white/pink (ticket mahbot-1025).
        let conditions: Vec<(&str, Vec<f32>)> = vec![
            ("clean", agc_audio.clone()),
            (
                "10db_white",
                add_noise(
                    &agc_audio,
                    10.0,
                    NoiseType::White,
                    Some(clip_idx as u64 + 1),
                ),
            ),
            (
                "20db_white",
                add_noise(
                    &agc_audio,
                    20.0,
                    NoiseType::White,
                    Some(clip_idx as u64 + 2),
                ),
            ),
            (
                "10db_pink",
                add_noise(&agc_audio, 10.0, NoiseType::Pink, Some(clip_idx as u64 + 3)),
            ),
            (
                "20db_pink",
                add_noise(&agc_audio, 20.0, NoiseType::Pink, Some(clip_idx as u64 + 4)),
            ),
        ];
        for (variant_idx, (cond, audio)) in conditions.iter().enumerate() {
            match super::process_enrollment_sample(audio) {
                Ok(embs) if !embs.is_empty() => {
                    let n = embs.len();
                    sequences.push(EmbeddingSequence::negative(
                        UtteranceId {
                            sequence_index: clip_idx,
                            variant_index: variant_idx,
                        },
                        Source::CrossSpeaker,
                        None,
                        embs,
                    ));
                    info!("Cross-speaker negative '{label}' {cond}: {n} embeddings (mahbot-1025)",);
                }
                _ => warn!("Cross-speaker negative '{label}' {cond}: no embeddings"),
            }
        }
    }
    sequences
}

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
#[allow(clippy::struct_excessive_bools)]
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

    // ── mahbot-1012 per-frame scoring geometry (parallel to per_frame_scores)
    /// Stable hash of each scored embedding (FNV-1a over f32 bit patterns).
    per_frame_embedding_hashes: Vec<u64>,
    /// L2 norm of each scored embedding.
    per_frame_embedding_l2_norms: Vec<f32>,
    /// Mel-frame position where each scored window started.
    per_frame_window_start: Vec<usize>,
    /// Mel buffer length at each scoring step.
    per_frame_mel_buffer_len: Vec<usize>,
    /// Geometry class of each scored window (see
    /// [`super::WindowGeometry`]).
    per_frame_geometry: Vec<super::WindowGeometry>,
    /// Adaptive-threshold mode of each scored frame (see
    /// [`super::AdaptiveFrameMode`]).
    per_frame_adaptive_mode: Vec<super::AdaptiveFrameMode>,
    /// Candidate lifecycle state after each scored frame (see
    /// [`super::CandidateFrameState`]).
    per_frame_candidate_state: Vec<super::CandidateFrameState>,
    /// Per-hop VAD decisions during streaming detection (one per VAD decision
    /// in processing order).
    per_hop_vad: Vec<bool>,

    // ── mahbot-1023 deferred-burst / acceptance-floor fields ──────────────
    /// Scoring path that produced the detection (raw source: "burst" /
    /// "segment_end_pass" / "other").  `None` when not detected.  The
    /// enrolled-phase report reclassifies "other" via
    /// [`candidate_created_path`](Self::candidate_created_path): a
    /// main-loop confirmation of a burst-created candidate is the PRIMARY
    /// mechanism ("burst_continuation"); a main-loop detection whose
    /// candidate the main loop created itself is "unexpected" (mahbot-1024).
    detection_path: Option<String>,
    /// Scoring path that created the ACTIVE classifier candidate ("burst" /
    /// "segment_end_pass" / "other"; `None` when no candidate is active).
    /// Evidence for the [`detection_path`](Self::detection_path)
    /// reclassification (mahbot-1024).
    candidate_created_path: Option<String>,
    /// Whether the deferred burst sweep ran during this variant's session.
    burst_sweep_fired: bool,
    /// Mel-frame buffer length at burst-sweep time (the actual live trigger
    /// geometry — typically 68–80).
    burst_sweep_buffer_len: Option<usize>,
    /// Whether the segment-end pass ran at a boundary during this variant's
    /// session.
    segment_end_pass_fired: bool,
    /// Whether the adaptive threshold was still bootstrapping at the end of
    /// the test utterance (mahbot-1023 score-only feed rule: a full
    /// high-scoring utterance can keep the bootstrap alive for its whole
    /// duration — the effective threshold then stays at the static hard
    /// floor, and miss verdicts must not misread this as an adaptive block).
    adaptive_bootstrap_persisted: bool,
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
        // mahbot-1012 per-frame geometry (copied from instrumentation; the
        // enum arrays are cloned as-is — JSON conversion happens in pv_to_json).
        per_frame_embedding_hashes: ctx.instrumentation.per_frame_embedding_hashes.clone(),
        per_frame_embedding_l2_norms: ctx.instrumentation.per_frame_embedding_l2_norms.clone(),
        per_frame_window_start: ctx.instrumentation.per_frame_window_start.clone(),
        per_frame_mel_buffer_len: ctx.instrumentation.per_frame_mel_buffer_len.clone(),
        per_frame_geometry: ctx.instrumentation.per_frame_geometry.clone(),
        per_frame_adaptive_mode: ctx.instrumentation.per_frame_adaptive_mode.clone(),
        per_frame_candidate_state: ctx.instrumentation.per_frame_candidate_state.clone(),
        per_hop_vad: ctx.instrumentation.per_hop_vad.clone(),
        // mahbot-1023: deferred-burst / acceptance-floor evidence.
        detection_path: ctx.instrumentation.detection_path.map(str::to_string),
        candidate_created_path: ctx
            .instrumentation
            .candidate_created_path
            .map(str::to_string),
        burst_sweep_fired: ctx.instrumentation.burst_sweep_fired,
        burst_sweep_buffer_len: ctx.instrumentation.burst_sweep_buffer_len,
        segment_end_pass_fired: ctx.instrumentation.segment_end_pass_fired,
        adaptive_bootstrap_persisted: ctx
            .instrumentation
            .per_frame_adaptive_mode
            .last()
            .is_some_and(|m| *m == super::AdaptiveFrameMode::Bootstrap),
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
    /// The first classifier threshold crossing fell inside the verifier
    /// warm-up epoch (frame-accurate, mahbot-1024: frame `i` is warm-up iff
    /// `warmup_n_embeddings + i + 1 < VERIFIER_WARMUP_EMBEDDINGS`); the
    /// trigger was suppressed by verifier warm-up
    /// ([`VERIFIER_WARMUP_EMBEDDINGS`]).
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
///
/// ## Warm-up suppression is frame-accurate (mahbot-1024)
///
/// The pre-utterance `warmup_completed` flag is captured BEFORE the
/// cold-start utterance (always `false` in the cold pass — the utterance
/// itself consumes the verifier warm-up), so it cannot distinguish a
/// trigger consumed by warm-up suppression from a genuine post-warm-up
/// verifier rejection.  The mahbot-1023 speed_down miss (run
/// 20260801-061348) was mislabeled `warmup_suppression`: its first
/// threshold crossing was frame 3 — the exact warm-up boundary — and the
/// verifier genuinely rejected it (peak 0.7886 < 0.86).  A second observed
/// speed_down sub-floor peak is 0.8090 (run 20260801-085648) — the two
/// readings are the archive's only speed_down sub-floor peaks (mahbot-1025
/// doc fix; the fresh-run peaks 0.9485 / 0.8983 / 0.8620 all cleared the
/// floor).  The verdict now
/// uses [`trigger_fell_in_warmup`]: frame `i` is a warm-up frame iff
/// `warmup_n_embeddings + i + 1 < VERIFIER_WARMUP_EMBEDDINGS` (the ring is
/// pushed before the warm-up test).
fn classify_miss(pv: &PerVariantResult) -> MissVerdict {
    if pv.detected {
        return MissVerdict::Detected;
    }
    if pv.vad_speech_frames == 0 || pv.per_frame_scores.is_empty() {
        return MissVerdict::VadFailure;
    }
    let triggered = pv.max_rolling_sum >= MIN_CLASSIFIER_THRESHOLD;
    if triggered && !pv.warmup_completed && trigger_fell_in_warmup(pv) {
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
        // ── mahbot-1023 prolonged-bootstrap branch ──
        // With the score-only bootstrap feed rule, a full high-scoring
        // utterance can keep the adaptive state in bootstrap for its whole
        // duration.  During bootstrap the effective threshold IS the static
        // hard floor (peek returns None → match_threshold), so reaching the
        // hard floor (`triggered`) implies crossing the effective threshold —
        // this arm is structurally UNREACHABLE under a persisted bootstrap.
        // It is kept as an explicit branch so report readers cannot misread a
        // persisted bootstrap as an adaptive-threshold failure: the
        // `adaptive_bootstrap_persisted` per-variant field makes the state
        // explicit, and the verifier split below is the accurate attribution
        // whenever the classifier triggered.
        return MissVerdict::AdaptiveThresholdBlocked;
    }
    if !pv.verifier_trained {
        // No second-stage gate: a miss after crossing the effective threshold
        // is a candidate lifecycle/timing issue (expired or audio ended).
        return MissVerdict::VerifierTiming;
    }
    // Split by the CONSTANT acceptance floor (0.86, mahbot-1023): the
    // runtime-calibrated `pv.verifier_threshold` is a report-only reference
    // and does not arbitrate acceptance.
    if pv.verifier_score >= crate::audio::voice_verifier::VERIFIER_ACCEPTANCE_FLOOR {
        MissVerdict::VerifierTiming
    } else {
        MissVerdict::VerifierRejected
    }
}

/// Frame-accurate warm-up-trigger test (mahbot-1024).
///
/// Returns `true` when the first classifier threshold crossing fell inside
/// the verifier warm-up epoch.  `per_frame_scores[i]` is the
/// `(warmup_n_embeddings + i + 1)`-th embedding pushed to the ring (the
/// ring is pushed before the warm-up test in `score_single_embedding`), so
/// frame `i` is a warm-up frame iff
/// `warmup_n_embeddings + i + 1 < VERIFIER_WARMUP_EMBEDDINGS`.
fn trigger_fell_in_warmup(pv: &PerVariantResult) -> bool {
    let warmup_frames_remaining = crate::audio::voice_verifier::VERIFIER_WARMUP_EMBEDDINGS
        .saturating_sub(pv.warmup_n_embeddings + 1);
    pv.first_trigger_frame_idx
        .is_some_and(|i| i < warmup_frames_remaining)
}

/// Reclassify the raw detection source into the enrolled-phase mechanism
/// taxonomy (mahbot-1024 re-scope).
///
/// The raw `detection_source` is the window type where the confirmation
/// fired.  The measured production mechanism is the **burst-created
/// candidate confirmed mid-utterance by main-loop continuation** (12/14 of
/// the archived final-code detections): the deferred burst sweep creates
/// the candidate, and a subsequent clean main-loop (stride-8) window
/// confirms it at verifier ≥ 0.86 before the segment-end pass, 760–790 ms
/// from wake-word onset.  That path must NOT be flagged as a regression.
///
/// Returned labels:
/// - `"burst"` — the deferred burst sweep confirmed directly.
/// - `"burst_continuation"` — a burst-created candidate was confirmed by a
///   later main-loop window (the PRIMARY expected mechanism).
/// - `"segment_end_pass"` — the segment-boundary fallback pass confirmed.
/// - `"unexpected"` — a main-loop detection whose candidate the main loop
///   created itself (no burst-created candidate to continue) — the genuine
///   unexpected-path flag.
///
/// `None` when the variant was not detected.
fn reclassify_detection_path(
    detection_source: Option<&str>,
    candidate_created_path: Option<&str>,
) -> Option<String> {
    let mechanism = match detection_source? {
        "burst" | "segment_end_pass" => detection_source?,
        "other" => {
            if candidate_created_path == Some("burst") {
                "burst_continuation"
            } else {
                "unexpected"
            }
        }
        other => other,
    };
    Some(mechanism.to_string())
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
/// | Triggered + warm-up-epoch trigger (frame-accurate, mahbot-1024) | `warmup_suppressed` |
/// | Triggered + post-warm-up trigger + `detected == false` | `verifier_caught` |
/// | Triggered + `detected == true` | `full_pipeline_fa` |
#[derive(Debug, Default, Clone, Copy)]
struct ClassifierTriggerMetrics {
    /// Total variants tested in this group.
    total_variants: usize,
    /// Number of variants where max_rolling_sum >= MIN_CLASSIFIER_THRESHOLD.
    classifier_triggers: usize,
    /// Number where the first classifier trigger fell inside the verifier
    /// warm-up epoch (frame-accurate, [`trigger_fell_in_warmup`] — the
    /// pre-utterance `warmup_completed` flag alone would mislabel
    /// post-warm-up verifier rejections in the cold pass as warm-up
    /// suppression, mahbot-1024), so the verifier never evaluated these
    /// frames.
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
            // mahbot-1024: warm-up suppression is frame-accurate (the
            // pre-utterance warmup_completed flag is always false in the
            // cold pass and cannot distinguish warm-up-epoch triggers from
            // post-warm-up verifier rejections).
            if !pv.warmup_completed && trigger_fell_in_warmup(pv) {
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
        // Constant acceptance floor (0.86, mahbot-1023) — the runtime
        // calibrated value is report-only.
        if pv.verifier_score >= crate::audio::voice_verifier::VERIFIER_ACCEPTANCE_FLOOR {
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
/// the exhaustive miss verdict, rejection-margin evidence, and the mahbot-1012
/// exactly-one-of-five localization bucket are added.
// Clippy: this is a pure JSON serialization shim — one insert per field; the
// line count is inherent to the schema, not control flow worth splitting.
#[expect(clippy::too_many_lines)]
fn pv_to_json(
    pv: &PerVariantResult,
    category: Option<&str>,
    localization: Option<&LocalizationRow>,
) -> serde_json::Value {
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
    // mahbot-1012: per-frame scoring geometry — parallel to per_frame_scores.
    obj.insert(
        "per_frame_embedding_hashes".to_string(),
        serde_json::json!(pv.per_frame_embedding_hashes),
    );
    obj.insert(
        "per_frame_embedding_l2_norms".to_string(),
        serde_json::json!(pv.per_frame_embedding_l2_norms),
    );
    obj.insert(
        "per_frame_window_start".to_string(),
        serde_json::json!(pv.per_frame_window_start),
    );
    obj.insert(
        "per_frame_mel_buffer_len".to_string(),
        serde_json::json!(pv.per_frame_mel_buffer_len),
    );
    obj.insert(
        "per_frame_geometry".to_string(),
        serde_json::json!(
            pv.per_frame_geometry
                .iter()
                .map(|g| g.as_str())
                .collect::<Vec<_>>()
        ),
    );
    obj.insert(
        "per_frame_adaptive_mode".to_string(),
        serde_json::json!(
            pv.per_frame_adaptive_mode
                .iter()
                .map(|m| m.as_str())
                .collect::<Vec<_>>()
        ),
    );
    obj.insert(
        "per_frame_candidate_state".to_string(),
        serde_json::json!(
            pv.per_frame_candidate_state
                .iter()
                .map(|c| c.as_str())
                .collect::<Vec<_>>()
        ),
    );
    obj.insert("per_hop_vad".to_string(), serde_json::json!(pv.per_hop_vad));
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
    // mahbot-1023: deferred-burst / acceptance-floor evidence (all variants).
    obj.insert(
        "detection_path".to_string(),
        serde_json::json!(pv.detection_path),
    );
    // mahbot-1024: reclassified mechanism taxonomy (see
    // reclassify_detection_path).  The raw detection_path stays as evidence;
    // "mechanism" is the report's taxonomy label.
    obj.insert(
        "mechanism".to_string(),
        serde_json::json!(reclassify_detection_path(
            pv.detection_path.as_deref(),
            pv.candidate_created_path.as_deref()
        )),
    );
    obj.insert(
        "candidate_created_path".to_string(),
        serde_json::json!(pv.candidate_created_path),
    );
    obj.insert(
        "burst_sweep_fired".to_string(),
        serde_json::json!(pv.burst_sweep_fired),
    );
    obj.insert(
        "burst_sweep_buffer_len".to_string(),
        serde_json::json!(pv.burst_sweep_buffer_len),
    );
    obj.insert(
        "segment_end_pass_fired".to_string(),
        serde_json::json!(pv.segment_end_pass_fired),
    );
    obj.insert(
        "adaptive_bootstrap_persisted".to_string(),
        serde_json::json!(pv.adaptive_bootstrap_persisted),
    );
    obj.insert(
        "effective_verifier_threshold".to_string(),
        serde_json::json!(crate::audio::voice_verifier::VERIFIER_ACCEPTANCE_FLOOR),
    );
    // Positive-only miss evidence (mahbot-1005 §2/§3/§4, mahbot-1012 §4).
    if category.is_none() {
        let verdict = classify_miss(pv);
        obj.insert("verdict".to_string(), serde_json::json!(verdict.as_str()));
        // mahbot-1012: exactly-one-of-five localization bucket (with the
        // supporting evidence trace) for this (variant, pass).
        if let Some(row) = localization {
            obj.insert(
                "localization_bucket".to_string(),
                serde_json::json!(row.bucket.as_str()),
            );
            obj.insert("localization_evidence".to_string(), row.evidence.clone());
        }
        if pv.verifier_trained && !pv.detected {
            // Margin to the CONSTANT acceptance floor (0.86, mahbot-1023) —
            // the gate the acceptance protocol applies.  The runtime
            // calibrated threshold remains visible via verifier_threshold.
            obj.insert(
                "verifier_rejection_margin".to_string(),
                serde_json::json!(
                    (crate::audio::voice_verifier::VERIFIER_ACCEPTANCE_FLOOR - pv.verifier_score)
                        .max(0.0)
                ),
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
/// Cosine similarity for dual-path drift analysis (mahbot-1012 §2).
///
/// Deliberately does NOT delegate to [`crate::vector::cosine_similarity`]:
/// that helper clamps to [0, 1], which would erase negative cosines that are
/// meaningful drift evidence (anti-correlated embeddings between the two
/// paths).  The canonical helper's length/NaN guards are also skipped — both
/// sides here are same-dimension embedding vectors produced by the same model,
/// and a NaN/zero-denominator result correctly reads as "no meaningful
/// similarity" for the report.  If a future consolidation re-introduces the
/// clamp, negative-cosine evidence in the dual-path report would silently
/// disappear — keep this divergence documented.
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

// ── mahbot-1012: same-audio training-vs-streaming comparison ───────────────
// Measurement-only instrumentation: runs the SAME PCM through both the
// training/enrollment embedding path and the streaming detection path, records
// per-window embedding hashes + classifier/verifier scores on both paths, runs
// an anchor-comparison probe (training window grid vs natural streaming grid),
// and classifies every positive variant into exactly one of five localization
// buckets.  No production behavior changes — all data is report-only.

/// One side of a dual-path capture: window-by-window embedding + score data.
struct DualPathSide {
    /// Mel-frame position where each scored window started.
    window_start: Vec<usize>,
    /// Number of REAL mel frames inside each scored window (the rest is
    /// synthetic padding from the deferred-burst / segment-end-pass padded
    /// geometry).  Streaming side: the mel buffer length at scoring time
    /// minus the window start, capped at
    /// [`EMBEDDING_WINDOW_FRAMES`].  Training side: the whole-utterance mel
    /// length minus the window start, capped.  Quantifies the padded-window
    /// geometry divergence — a streaming window-0 scored at the burst trigger
    /// (~68–76 real frames at the live flush) vs a training window-0 padded
    /// from the full utterance, so the real-frame gap is much smaller than
    /// the old incremental short-buffer fallback's ~10-frame windows (removed
    /// in mahbot-1023).
    real_frames: Vec<usize>,
    /// Stable hash of each scored window's embedding.
    hashes: Vec<u64>,
    /// L2 norm of each scored window's embedding.
    embedding_l2: Vec<f32>,
    /// Per-window classifier score (sigmoid total_score).
    classifier_scores: Vec<f32>,
    /// Per-window verifier score (0.0 = not evaluated: verifier warm-up /
    /// untrained — NOT a zero-confidence rejection).
    verifier_scores: Vec<f32>,
    /// Maximum rolling sum (3-window decayed accumulator) achieved.
    max_rolling_sum: f32,
    /// Geometry class of each scored window (snake_case, streaming side only).
    geometry: Vec<&'static str>,
}

impl DualPathSide {
    fn empty() -> Self {
        Self {
            window_start: Vec::new(),
            real_frames: Vec::new(),
            hashes: Vec::new(),
            embedding_l2: Vec::new(),
            classifier_scores: Vec::new(),
            verifier_scores: Vec::new(),
            max_rolling_sum: 0.0,
            geometry: Vec::new(),
        }
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "n_windows": self.window_start.len(),
            "window_start": self.window_start,
            "real_frames_per_window": self.real_frames,
            "hashes": self.hashes,
            "embedding_l2": self.embedding_l2,
            "classifier_scores": self.classifier_scores,
            "verifier_scores": self.verifier_scores,
            "max_rolling_sum": self.max_rolling_sum,
            "geometry": self.geometry,
        })
    }
}

/// One ordinal window match between the streaming path and the training path.
///
/// Pairing rule (mahbot-1012 §2): **ordinal** — streaming window `i` is paired
/// with training window `i`, because both paths anchor their window grids at
/// speech onset (mel frame 0).  When the streaming path produces extra windows
/// (e.g. the deferred-burst / segment-end-pass padded windows), those have
/// `training_idx: None` and are reported as unmatched.  The per-window
/// `window_start` values are reported for both sides so grid drift is directly
/// visible.
///
/// # Pairing caveat: buffer-relative vs absolute starts (mahbot-1012 reviewer)
///
/// The streaming `window_start` is **buffer-relative**: the mel-buffer trim at
/// the end of `handle_wake_word_detection` decrements `next_window_start`, so
/// after a trim the streaming grid restarts near 0 even though the audio
/// continues — the streaming starts are indices into the CURRENT (trimmed)
/// buffer, not absolute positions from speech onset.  The training starts are
/// absolute `i * 8` positions over the whole-utterance mel.  For clips longer
/// than ~68 mel frames a trim can therefore make `start_delta` /
/// `first_mismatch_idx` read as grid drift or "divergence starts at window k"
/// when the cause is trim re-indexing.  The current benchmark clips are all
/// shorter than 76 mel frames (no trims), so this is latent here; the
/// per-window `per_frame_mel_buffer_len` in the per-variant instrumentation
/// makes the re-indexing reconstructible when it does occur.
struct DualPathWindowMatch {
    streaming_idx: usize,
    training_idx: Option<usize>,
    streaming_start: usize,
    training_start: Option<usize>,
    /// `streaming_start - training_start` (positive = streaming grid lags the
    /// training anchor).
    start_delta: Option<i64>,
    /// Number of REAL mel frames in the streaming window (rest is synthetic
    /// padding from the burst/pass padded geometry).  Window-0 carries
    /// ~68–76 real frames at the live burst trigger (vs ~57 for the training
    /// full-buffer pad) — the residual real-frame gap quantifies the
    /// padded-window geometry divergence that a raw hash/cosine comparison
    /// alone cannot separate from the mel-scope divergence.
    streaming_real_frames: usize,
    /// Number of REAL mel frames in the training window.
    training_real_frames: usize,
    /// Whether the two embeddings are bit-identical (hash equal).
    hash_match: Option<bool>,
    /// Cosine similarity of the two embeddings.
    cosine: Option<f32>,
    /// L2 distance between the two embeddings.
    l2_delta: Option<f32>,
    /// Streaming-side classifier score for this window.
    streaming_score: f32,
    /// Training-side classifier score for this window.
    training_score: Option<f32>,
    /// Streaming-side geometry class (snake_case).
    streaming_geometry: &'static str,
}

/// Dual-path same-audio comparison for one PCM clip (mahbot-1012 §2).
struct DualPathComparison {
    label: String,
    detected: bool,
    streaming: DualPathSide,
    training: DualPathSide,
    matches: Vec<DualPathWindowMatch>,
    /// Smallest ordinal window index whose hash mismatches its training
    /// counterpart.  `None` when every paired window matches (or no windows).
    first_mismatch_idx: Option<usize>,
    /// Fraction of ordinal-paired windows whose hashes match (0..1).
    hash_match_frac: f64,
    /// Cosine similarity at the first ordinal pair (window 0) — the earliest
    /// point where mel-value divergence can be detected.
    window0_cosine: Option<f32>,
    /// L2 delta at the first ordinal pair (window 0).
    window0_l2_delta: Option<f32>,
}

impl DualPathComparison {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "variant": self.label,
            "detected": self.detected,
            "streaming": self.streaming.to_json(),
            "training": self.training.to_json(),
            "matches": self.matches.iter().map(|m| serde_json::json!({
                "streaming_idx": m.streaming_idx,
                "training_idx": m.training_idx,
                "streaming_start": m.streaming_start,
                "training_start": m.training_start,
                "start_delta": m.start_delta,
                "hash_match": m.hash_match,
                "cosine": m.cosine,
                "l2_delta": m.l2_delta,
                "streaming_score": m.streaming_score,
                "training_score": m.training_score,
                "streaming_real_frames": m.streaming_real_frames,
                "training_real_frames": m.training_real_frames,
                "streaming_geometry": m.streaming_geometry,
            })).collect::<Vec<_>>(),
            "first_mismatch_idx": self.first_mismatch_idx,
            "hash_match_frac": self.hash_match_frac,
            "window0_cosine": self.window0_cosine,
            "window0_l2_delta": self.window0_l2_delta,
        })
    }
}

/// Raw-mel comparison between the two paths for one representative variant
/// (mahbot-1012 §2).  Both paths consume the SAME AGC'd PCM (fresh
/// preprocessor): the training side runs the whole-utterance mel call, the
/// streaming side runs the VAD-gated per-batch mel calls.  The manager's
/// per-call dynamic-range floor hypothesis predicts the streaming frames'
/// global min/max and per-frame norms differ from the training frames' even
/// though the audio is identical.
struct MelFrameComparison {
    variant: String,
    training_n_frames: usize,
    streaming_n_frames: usize,
    training_global_min: f32,
    training_global_max: f32,
    streaming_global_min: f32,
    streaming_global_max: f32,
    /// Per-frame L2 norm of the training mel frames (compact comparison).
    training_frame_norms: Vec<f32>,
    /// Per-frame L2 norm of the streaming mel frames.
    streaming_frame_norms: Vec<f32>,
    /// First 24 mel frames of each path (32 mel bands each) for direct
    /// inspection of the normalization scope.
    training_first_frames: Vec<Vec<f32>>,
    streaming_first_frames: Vec<Vec<f32>>,
}

impl MelFrameComparison {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "variant": self.variant,
            "note": "Both paths consume the same AGC'd PCM; the training side \
                     runs one whole-utterance mel ONNX call, the streaming side \
                     runs VAD-gated per-batch mel ONNX calls.  If the mel model \
                     applies a per-call global-max dynamic-range floor, the \
                     streaming global min/max and per-frame norms will differ \
                     from the training values on identical audio.",
            "training": {
                "n_frames": self.training_n_frames,
                "global_min": self.training_global_min,
                "global_max": self.training_global_max,
                "per_frame_l2_norms": self.training_frame_norms,
                "first_24_frames": self.training_first_frames,
            },
            "streaming": {
                "n_frames": self.streaming_n_frames,
                "global_min": self.streaming_global_min,
                "global_max": self.streaming_global_max,
                "per_frame_l2_norms": self.streaming_frame_norms,
                "first_24_frames": self.streaming_first_frames,
            },
        })
    }
}

/// Anchor-comparison probe result (mahbot-1012 §3): the same audio scored
/// with the natural streaming window grid vs the training-anchored grid
/// (stride 8 from mel frame 0).  The anchored side reuses the streaming mel
/// layout (VAD-gated per-batch mel calls) so the mel-scope divergence is held
/// constant.
///
/// # What this probe actually isolates (mahbot-1012 reviewer)
///
/// For utterances shorter than [`EMBEDDING_WINDOW_FRAMES`] mel frames — the
/// entire current benchmark — both grids start at mel frame 0, so there is no
/// grid drift to measure: the natural streaming grid IS the training anchor.
/// The natural/anchored score gap on these clips is driven by **window
/// content**: the streaming side's padded windows (deferred burst /
/// segment-end pass) contain only the mel frames accumulated at scoring time
/// (e.g. ~68–76 real frames at the live burst trigger, the rest synthetic),
/// while the training path scores ONE full-buffer padded window (~57 real
/// frames).  The per-window `real_frames` fields make this explicit; a large
/// natural/anchored gap with a large real-frame gap is evidence for the
/// padded-window geometry divergence (2), not grid drift (3).  Grid drift
/// would require clips long enough for mel-buffer trims / gap-recovery
/// re-anchoring to shift the grid off the anchor.
struct AnchorProbeResult {
    variant: String,
    natural_starts: Vec<usize>,
    natural_scores: Vec<f32>,
    natural_max_rolling_sum: f32,
    /// Real (non-synthetic) mel frames in each natural-side window.
    natural_real_frames: Vec<usize>,
    anchored_starts: Vec<usize>,
    anchored_scores: Vec<f32>,
    anchored_max_rolling_sum: f32,
    /// Real (non-synthetic) mel frames in each anchored-side window.
    anchored_real_frames: Vec<usize>,
}

impl AnchorProbeResult {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "variant": self.variant,
            "note": "Both sides use the per-batch streaming mel layout (VAD-gated \
                     per-batch ONNX calls), so the mel-scope divergence is held \
                     constant.  For sub-76-frame utterances the natural grid starts \
                     at mel frame 0 — identical to the training anchor — so the \
                     natural/anchored score gap reflects window CONTENT \
                     (partial-buffer vs full-buffer padding; see real_frames), not \
                     grid drift.  Window real-frame counts quantify the padded-window \
                     geometry divergence (2).",
            "natural_streaming_grid": {
                "window_start": self.natural_starts,
                "real_frames": self.natural_real_frames,
                "classifier_scores": self.natural_scores,
                "max_rolling_sum": self.natural_max_rolling_sum,
            },
            "training_anchored_grid": {
                "window_start": self.anchored_starts,
                "real_frames": self.anchored_real_frames,
                "classifier_scores": self.anchored_scores,
                "max_rolling_sum": self.anchored_max_rolling_sum,
            },
        })
    }
}

/// Exactly-one-of-five localization bucket for a positive variant
/// (mahbot-1012 §4).  Precedence follows the ticket's causal chain —
/// (a) > (b) > (c) > (d) > (e) — with `Detected` first.  A variant can satisfy
/// multiple conditions (e.g. hash mismatch AND low scores); the highest
/// causal bucket wins so the report's bucket counts are exhaustive and
/// disjoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalizationBucket {
    Detected,
    /// (a) embeddings never reached the classifier (VAD no-speech or zero
    ///     scored windows).
    EmbeddingsNeverReachedClassifier,
    /// (b) embeddings differ from the training path (hash mismatch), with the
    ///     geometry fields showing where the divergence starts.
    EmbeddingsDiffer,
    /// (c) embeddings match the training path but classifier scores are low
    ///     (streaming mean < training-path in-sample P10).
    EmbeddingsMatchScoresLow,
    /// (d) scores are adequate but the rolling gate never accumulated
    ///     (max_rolling_sum < MIN_CLASSIFIER_THRESHOLD).
    ScoresAdequateGateNeverAccumulated,
    /// (e) the gate passed but the verifier stage rejected / timing-failed.
    GatePassedVerifierRejected,
}

impl LocalizationBucket {
    fn as_str(self) -> &'static str {
        match self {
            Self::Detected => "detected",
            Self::EmbeddingsNeverReachedClassifier => "embeddings_never_reached_classifier",
            Self::EmbeddingsDiffer => "embeddings_differ",
            Self::EmbeddingsMatchScoresLow => "embeddings_match_scores_low",
            Self::ScoresAdequateGateNeverAccumulated => "scores_adequate_gate_never_accumulated",
            Self::GatePassedVerifierRejected => "gate_passed_verifier_rejected",
        }
    }
}

/// One localized (variant, pass) row with its supporting evidence.
struct LocalizationRow {
    variant: String,
    /// "warm" or "cold".
    pass: &'static str,
    bucket: LocalizationBucket,
    evidence: serde_json::Value,
}

/// Shared parameter-passed embedding-scoring loop (mahbot-1022): scores a
/// sequence of embeddings through the classifier + verifier with a fresh
/// [`PipelineCtx`](super::PipelineCtx), passing the classifier/verifier as
/// explicit references.  Pure — does NOT touch global voice state, so both the
/// mahbot-1012 training-path scoring and the B-sweep training-path scoring
/// (which must not call `set_classifier_weights` / `set_verifier`) share it.
/// Returns `(per_window_classifier_scores, per_window_verifier_scores,
/// max_rolling_sum)`.
fn score_embedding_sequence_with(
    embeddings: &[Vec<f32>],
    classifier: &WakeWordClassifier,
    verifier: &VoiceVerifier,
) -> (Vec<f32>, Vec<f32>, f32) {
    let mut ctx = super::PipelineCtx::new();
    let mut scores = Vec::with_capacity(embeddings.len());
    let mut verifier_scores = Vec::with_capacity(embeddings.len());
    let mut max_rs = 0.0f32;
    for emb in embeddings {
        let (_, rolling_sum, total_score, _, max_verifier, _) = super::score_single_embedding(
            emb,
            &mut ctx.embedding_ring,
            Some(classifier),
            Some(verifier),
            &mut ctx.score_window,
            None, // static threshold — the training path has no adaptive state
            ctx.adaptive_k,
            None, // no candidate tracking on the training side
        );
        scores.push(total_score);
        verifier_scores.push(max_verifier);
        max_rs = max_rs.max(rolling_sum);
    }
    (scores, verifier_scores, max_rs)
}

/// Score a sequence of embeddings through the classifier + verifier with a
/// fresh [`PipelineCtx`](super::PipelineCtx) (no AGC/VAD/mel — pure embedding
/// scoring, the same `score_single_embedding` call the enrollment self-test
/// uses).  Mirrors [`score_embedding_sequence_with`] but first re-asserts the
/// trained weights into the global voice state (the mahbot-1012 comparison's
/// documented behavior).  Returns `(per_window_classifier_scores,
/// per_window_verifier_scores, max_rolling_sum)`.
fn score_embedding_sequence(
    embeddings: &[Vec<f32>],
    classifier: &WakeWordClassifier,
    verifier: &VoiceVerifier,
) -> (Vec<f32>, Vec<f32>, f32) {
    super::set_classifier_weights(classifier.weights_ref().clone());
    super::set_verifier(verifier.clone());
    score_embedding_sequence_with(embeddings, classifier, verifier)
}

/// AGC a PCM clip through a fresh AudioPreprocessor (the training path's
/// preprocessing — same CONFIG-driven config as streaming, fresh state).
fn agc_pcm(pcm: &[f32]) -> Vec<f32> {
    use crate::audio::audio_preprocessor::AudioPreprocessor;
    let mut pre = AudioPreprocessor::new(enrollment_preprocessor_config());
    let mut agc_audio: Vec<f32> = Vec::with_capacity(pcm.len());
    for chunk in pcm.chunks(super::FRAME_LENGTH) {
        agc_audio.extend(pre.process(chunk.to_vec()));
    }
    agc_audio
}

/// Run the SAME PCM through both paths and compare per-window embeddings and
/// scores (mahbot-1012 §2).
///
/// # Same-PCM semantics
/// The streaming side feeds the raw PCM through [`run_streaming_detection`]
/// with a fresh [`PipelineCtx`](super::PipelineCtx) (fresh AGC/NS, cold pass —
/// no warm-up), exactly like production's post-silence segment start.  The
/// training side AGCs the same raw PCM through a fresh AudioPreprocessor
/// (same CONFIG-driven config), VAD-gates it with the same threshold and a
/// fresh detector (production enrollment VAD-segments utterances before
/// embedding extraction), and runs the enrollment embedding path (one
/// whole-utterance mel call on the VAD-gated speech + stride-8 windows).
/// Both sides therefore start from the same raw PCM, the same fresh AGC
/// config, and the same VAD gate; the remaining differences are the mel-call
/// granularity (per-batch vs whole-utterance) and the window grid — which is
/// precisely what this ticket measures.
// Clippy: the dual-path harness is a single linear capture + pairing pipeline
// (131 lines in the voice-tests configuration); splitting it would obscure the
// ordinal-pairing logic it exists to make auditable.
#[expect(clippy::too_many_lines)]
fn run_dual_path_capture(
    pcm: &[f32],
    label: &str,
    classifier: &WakeWordClassifier,
    verifier: &VoiceVerifier,
) -> DualPathComparison {
    use crate::audio::audio_preprocessor::AudioPreprocessor;
    let models = super::ONNX_MODELS
        .get()
        .expect("ONNX models must be loaded by the benchmark");

    // ── Streaming side (cold pass — fresh AGC, no warm-up) ────────────────
    let mut ctx = super::PipelineCtx::new();
    let result = run_streaming_detection(pcm, &mut ctx);
    let instr = &ctx.instrumentation;
    let streaming = DualPathSide {
        window_start: instr.per_frame_window_start.clone(),
        real_frames: instr
            .per_frame_mel_buffer_len
            .iter()
            .zip(instr.per_frame_window_start.iter())
            .map(|(&len, &start)| {
                len.saturating_sub(start)
                    .min(super::EMBEDDING_WINDOW_FRAMES)
            })
            .collect(),
        hashes: instr.per_frame_embedding_hashes.clone(),
        embedding_l2: instr.per_frame_embedding_l2_norms.clone(),
        classifier_scores: instr.per_frame_scores.iter().map(|s| s[0]).collect(),
        verifier_scores: instr.per_frame_verifier_scores.clone(),
        max_rolling_sum: max_rolling_sum(&instr.per_frame_scores),
        geometry: instr
            .per_frame_geometry
            .iter()
            .map(|g| g.as_str())
            .collect(),
    };

    // ── Training side (fresh AGC + VAD-gated enrollment embedding path) ────
    // Production enrollment VAD-segments each utterance BEFORE computing mel
    // frames, so training mel frame 0 = speech onset (mahbot-1001 Fix 3:
    // context_padding_samples = 0).  The harness mirrors this by VAD-gating
    // the AGC'd PCM (fresh detector, same gate as the streaming path) and
    // running ONE whole-utterance mel call on the resulting speech audio —
    // the training path's mel scope.  The remaining difference vs the
    // streaming side is therefore the mel-call granularity (one whole-
    // utterance call vs per-batch flushes) and the window grid, which is
    // exactly what the ticket measures.
    let mut pre = AudioPreprocessor::new(enrollment_preprocessor_config());
    let mut agc_audio: Vec<f32> = Vec::with_capacity(pcm.len());
    for chunk in pcm.chunks(super::FRAME_LENGTH) {
        agc_audio.extend(pre.process(chunk.to_vec()));
    }
    let (training, _training_starts, training_embeddings) = if agc_audio.is_empty() {
        (DualPathSide::empty(), Vec::new(), Vec::new())
    } else {
        let mut detector = Detector::default();
        let (_streaming_mel, speech_audio) = super::vad_gate_streaming_mel(&agc_audio, |hop| {
            super::is_speech_with_detector(hop, &mut detector, super::VAD_THRESHOLD)
        });
        if speech_audio.is_empty() {
            (DualPathSide::empty(), Vec::new(), Vec::new())
        } else {
            // Whole-utterance mel call (the training side's mel scope).
            let mel_frames =
                super::compute_mel_spectrogram(models, &speech_audio).unwrap_or_default();
            let embeddings =
                super::embeddings_from_mel_frames(models, &mel_frames).unwrap_or_default();
            // Training window grid: stride 8 from mel frame 0; a single padded
            // window at position 0 for utterances shorter than 76 mel frames.
            let starts = if mel_frames.len() < super::EMBEDDING_WINDOW_FRAMES {
                vec![0usize]
            } else {
                (0..embeddings.len()).map(|i| i * 8).collect()
            };
            let (scores, v_scores, max_rs) =
                score_embedding_sequence(&embeddings, classifier, verifier);
            let training_mel_len = mel_frames.len();
            (
                DualPathSide {
                    window_start: starts.clone(),
                    real_frames: starts
                        .iter()
                        .map(|&s| {
                            training_mel_len
                                .saturating_sub(s)
                                .min(super::EMBEDDING_WINDOW_FRAMES)
                        })
                        .collect(),
                    hashes: embeddings
                        .iter()
                        .map(|e| super::embedding_hash(e))
                        .collect(),
                    embedding_l2: embeddings
                        .iter()
                        .map(|e| super::embedding_l2_norm(e))
                        .collect(),
                    classifier_scores: scores,
                    verifier_scores: v_scores,
                    max_rolling_sum: max_rs,
                    geometry: Vec::new(), // training side has no streaming geometry
                },
                starts,
                embeddings,
            )
        }
    };

    // ── Ordinal pairing ───────────────────────────────────────────────────
    // Pair streaming window `i` with training window `i` (both paths anchor at
    // speech onset).  Extra streaming windows (burst/pass padded windows) have
    // no training counterpart and are reported as unmatched.  Cosine / L2 deltas
    // are computed from the RAW embeddings (streaming instrumentation + local
    // training embeddings) so the report quantifies drift, not just equality.
    let n_paired = streaming
        .window_start
        .len()
        .min(training.window_start.len());
    let mut matches = Vec::with_capacity(streaming.window_start.len());
    let mut n_hash_matches = 0usize;
    let mut first_mismatch_idx: Option<usize> = None;
    for i in 0..streaming.window_start.len() {
        let training_idx = if i < training.window_start.len() {
            Some(i)
        } else {
            None
        };
        let (hash_match, cosine, l2_delta, training_score) = match training_idx {
            Some(ti) => {
                let sh = streaming.hashes.get(i).copied();
                let th = training.hashes.get(ti).copied();
                let hash_equal = sh.is_some() && th.is_some() && sh == th;
                if i < n_paired {
                    if hash_equal {
                        n_hash_matches += 1;
                    } else if first_mismatch_idx.is_none() {
                        first_mismatch_idx = Some(i);
                    }
                }
                let (cos, l2) = if i < n_paired {
                    let s_emb = instr.per_frame_embeddings.get(i);
                    let t_emb = training_embeddings.get(ti);
                    match (s_emb, t_emb) {
                        (Some(s), Some(t)) => {
                            let cos = cosine_similarity(s, t);
                            let l2 = s
                                .iter()
                                .zip(t)
                                .map(|(a, b)| (a - b) * (a - b))
                                .sum::<f32>()
                                .sqrt();
                            (Some(cos), Some(l2))
                        }
                        _ => (None, None),
                    }
                } else {
                    (None, None)
                };
                (
                    Some(hash_equal),
                    cos,
                    l2,
                    training.classifier_scores.get(ti).copied(),
                )
            }
            None => (None, None, None, None),
        };
        matches.push(DualPathWindowMatch {
            streaming_idx: i,
            training_idx,
            // i < streaming.window_start.len() by loop construction, so direct
            // indexing cannot fail (reviewer feedback, mahbot-1012).
            streaming_start: streaming.window_start[i],
            training_start: training_idx.map(|ti| training.window_start[ti]),
            // usize → i64 via try_from (house style, cf. voice.rs stride-8
            // elapsed_ns conversion); window starts are tiny so the fallback
            // is unreachable in practice.
            start_delta: training_idx.map(|ti| {
                i64::try_from(streaming.window_start[i]).unwrap_or(i64::MAX)
                    - i64::try_from(training.window_start[ti]).unwrap_or(i64::MAX)
            }),
            streaming_real_frames: streaming.real_frames.get(i).copied().unwrap_or(0),
            training_real_frames: training_idx
                .and_then(|ti| training.real_frames.get(ti).copied())
                .unwrap_or(0),
            hash_match,
            cosine,
            l2_delta,
            streaming_score: streaming.classifier_scores.get(i).copied().unwrap_or(0.0),
            training_score,
            streaming_geometry: streaming.geometry.get(i).copied().unwrap_or(""),
        });
    }
    let hash_match_frac = if n_paired > 0 {
        n_hash_matches as f64 / n_paired as f64
    } else {
        0.0
    };
    let window0_cosine = matches.first().and_then(|m| m.cosine);
    let window0_l2_delta = matches.first().and_then(|m| m.l2_delta);

    DualPathComparison {
        label: label.to_string(),
        detected: result.detected,
        streaming,
        training,
        matches,
        first_mismatch_idx,
        hash_match_frac,
        window0_cosine,
        window0_l2_delta,
    }
}

/// Compare raw mel frames between the two paths for one representative variant
/// (mahbot-1012 §2 — settles the mel model's normalization-scope question:
/// per-call global max vs frequency-axis).  Both paths consume the SAME AGC'd
/// PCM (fresh preprocessor) and the SAME VAD gate (fresh detector): the
/// training side runs ONE whole-utterance mel ONNX call on the VAD-gated
/// speech audio, the streaming side runs VAD-gated per-batch mel ONNX calls.
/// Both are anchored at speech onset, so the ONLY difference is the mel-call
/// granularity (one call vs per-batch flushes).  If the mel model applies a
/// per-call global-max dynamic-range floor, the streaming frames' global
/// min/max and per-frame norms differ from the training frames' even though
/// the audio is identical.
fn run_mel_frame_comparison(pcm: &[f32], label: &str) -> Option<MelFrameComparison> {
    let models = super::ONNX_MODELS
        .get()
        .expect("ONNX models must be loaded by the benchmark");

    let agc_audio = agc_pcm(pcm);
    if agc_audio.is_empty() {
        return None;
    }

    // Shared VAD gate: both sides consume the same speech-anchored audio.
    let mut detector = Detector::default();
    let (streaming_mel, speech_audio) = super::vad_gate_streaming_mel(&agc_audio, |hop| {
        super::is_speech_with_detector(hop, &mut detector, super::VAD_THRESHOLD)
    });
    if speech_audio.is_empty() || streaming_mel.is_empty() {
        return None;
    }
    // Training side: one whole-utterance mel call on the VAD-gated speech.
    let training_mel = super::compute_mel_spectrogram(models, &speech_audio).ok()?;
    if training_mel.is_empty() {
        return None;
    }

    let frame_norms = |frames: &[Vec<f32>]| -> Vec<f32> {
        frames
            .iter()
            .map(|f| f.iter().map(|v| v * v).sum::<f32>().sqrt())
            .collect()
    };
    let global_min = |frames: &[Vec<f32>]| -> f32 {
        frames
            .iter()
            .flatten()
            .copied()
            .fold(f32::INFINITY, f32::min)
    };
    let global_max = |frames: &[Vec<f32>]| -> f32 {
        frames
            .iter()
            .flatten()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    };
    let first_frames =
        |frames: &[Vec<f32>]| -> Vec<Vec<f32>> { frames.iter().take(24).cloned().collect() };

    Some(MelFrameComparison {
        variant: label.to_string(),
        training_n_frames: training_mel.len(),
        streaming_n_frames: streaming_mel.len(),
        training_global_min: global_min(&training_mel),
        training_global_max: global_max(&training_mel),
        streaming_global_min: global_min(&streaming_mel),
        streaming_global_max: global_max(&streaming_mel),
        training_frame_norms: frame_norms(&training_mel),
        streaming_frame_norms: frame_norms(&streaming_mel),
        training_first_frames: first_frames(&training_mel),
        streaming_first_frames: first_frames(&streaming_mel),
    })
}

/// Anchor-comparison probe (mahbot-1012 §3): score the same audio with the
/// natural streaming window grid vs the training-anchored grid (stride 8 from
/// mel frame 0 = speech onset).  The anchored side reuses the streaming mel
/// layout (VAD-gated per-batch mel calls) so ONLY the window grid differs
/// from the natural streaming run — isolating divergence (3) from the
/// mel-scope divergence.
fn run_anchor_probe(
    pcm: &[f32],
    label: &str,
    classifier: &WakeWordClassifier,
    verifier: &VoiceVerifier,
) -> Option<AnchorProbeResult> {
    use crate::audio::audio_preprocessor::AudioPreprocessor;
    let models = super::ONNX_MODELS
        .get()
        .expect("ONNX models must be loaded by the benchmark");

    // ── Natural streaming grid ────────────────────────────────────────────
    let mut ctx = super::PipelineCtx::new();
    run_streaming_detection(pcm, &mut ctx);
    let natural_starts = ctx.instrumentation.per_frame_window_start.clone();
    let natural_scores: Vec<f32> = ctx
        .instrumentation
        .per_frame_scores
        .iter()
        .map(|s| s[0])
        .collect();
    let natural_real_frames: Vec<usize> = ctx
        .instrumentation
        .per_frame_mel_buffer_len
        .iter()
        .zip(natural_starts.iter())
        .map(|(&len, &start)| {
            len.saturating_sub(start)
                .min(super::EMBEDDING_WINDOW_FRAMES)
        })
        .collect();
    let natural_max_rolling_sum = max_rolling_sum(&ctx.instrumentation.per_frame_scores);
    if natural_starts.is_empty() {
        return None;
    }

    // ── Training-anchored grid over the streaming mel layout ──────────────
    let mut pre = AudioPreprocessor::new(enrollment_preprocessor_config());
    let mut agc_audio: Vec<f32> = Vec::with_capacity(pcm.len());
    for chunk in pcm.chunks(super::FRAME_LENGTH) {
        agc_audio.extend(pre.process(chunk.to_vec()));
    }
    let mut detector = Detector::default();
    let (mel_frames, _speech_audio) = super::vad_gate_streaming_mel(&agc_audio, |hop| {
        super::is_speech_with_detector(hop, &mut detector, super::VAD_THRESHOLD)
    });
    let embeddings = super::embeddings_from_mel_frames(models, &mel_frames).unwrap_or_default();
    if embeddings.is_empty() {
        return None;
    }
    let anchored_starts = if mel_frames.len() < super::EMBEDDING_WINDOW_FRAMES {
        vec![0usize]
    } else {
        (0..embeddings.len()).map(|i| i * 8).collect()
    };
    let anchored_real_frames: Vec<usize> = anchored_starts
        .iter()
        .map(|&start| {
            mel_frames
                .len()
                .saturating_sub(start)
                .min(super::EMBEDDING_WINDOW_FRAMES)
        })
        .collect();
    let (anchored_scores, _, anchored_max_rolling_sum) =
        score_embedding_sequence(&embeddings, classifier, verifier);

    Some(AnchorProbeResult {
        variant: label.to_string(),
        natural_starts,
        natural_scores,
        natural_max_rolling_sum,
        natural_real_frames,
        anchored_starts,
        anchored_scores,
        anchored_max_rolling_sum,
        anchored_real_frames,
    })
}

/// Classify a positive variant into exactly one of the five localization
/// buckets (mahbot-1012 §4).
///
/// ## Quantitative bucket boundaries
///
/// - (a) embeddings never reached the classifier: `vad_speech_frames == 0` or
///   zero scored windows on test-utterance frames.
/// - (b) embeddings differ: the first ordinal window's hash mismatches its
///   training counterpart OR fewer than 50% of ordinal-paired windows match.
///   A window-0 mismatch means the mel values diverged before any windowing /
///   grid effect (per-call mel floor or VAD gating); a later first-mismatch
///   with an overall low match fraction means the divergence starts at that
///   window (grid drift / padded-window geometry).
/// - (c) vs (d) boundary: "scores are low" is defined as the streaming
///   per-window mean total_score falling below the training-path in-sample
///   P10 (all training-side per-window scores across all positive variants).
///   When the training distribution is unavailable (empty captures), the
///   fallback boundary is [`NO_MATCH_RESET_THRESHOLD`] (0.316) — below that a
///   frame is treated as background, not speech, so the scores are clearly low.
/// - (e) everything that crossed the rolling gate but did not fire (verifier
///   rejection or verifier-timing; sub-annotated via the 8-way `verdict`).
///
/// ## AGC note
/// AGC non-convergence is a correlated diagnostic, not a causal bucket
/// (mahbot-1012): when `agc_converged == Some(false)` the evidence carries an
/// `agc_note` but the bucket assignment is unchanged.
// Clippy: the precedence chain is deliberately linear (exactly-one bucket,
// documented in the doc comment above); splitting it would hide the ordering.
#[expect(clippy::too_many_lines)]
fn classify_localization_bucket(
    pv: &PerVariantResult,
    comparison: &DualPathComparison,
    training_p10: Option<f32>,
) -> (LocalizationBucket, serde_json::Value) {
    let mut ev = serde_json::Map::new();
    let bucket = if pv.detected {
        LocalizationBucket::Detected
    } else if pv.vad_speech_frames == 0
        || pv.per_frame_scores.is_empty()
        || pv.n_test_embeddings == 0
    {
        // (a) embeddings never reached the classifier.
        ev.insert(
            "vad_speech_frames".to_string(),
            serde_json::json!(pv.vad_speech_frames),
        );
        ev.insert(
            "n_test_embeddings".to_string(),
            serde_json::json!(pv.n_test_embeddings),
        );
        LocalizationBucket::EmbeddingsNeverReachedClassifier
    } else {
        // Every positive variant has a dual-path comparison by construction
        // (comparisons_by_label is keyed by the same pos_variants labels that
        // produce the per-variant results), so this branch is unconditional —
        // there is no dual-path-unavailable fallback (mahbot-1012 reviewer).
        let cmp = comparison;
        ev.insert(
            "hash_match_frac".to_string(),
            serde_json::json!(cmp.hash_match_frac),
        );
        ev.insert(
            "window0_cosine".to_string(),
            serde_json::json!(cmp.window0_cosine),
        );
        ev.insert(
            "window0_l2_delta".to_string(),
            serde_json::json!(cmp.window0_l2_delta),
        );
        if let Some(idx) = cmp.first_mismatch_idx {
            ev.insert("first_mismatch_idx".to_string(), serde_json::json!(idx));
            if let Some(m) = cmp.matches.get(idx) {
                ev.insert(
                    "first_mismatch_streaming_start".to_string(),
                    serde_json::json!(m.streaming_start),
                );
                ev.insert(
                    "first_mismatch_training_start".to_string(),
                    serde_json::json!(m.training_start),
                );
                ev.insert(
                    "first_mismatch_geometry".to_string(),
                    serde_json::json!(m.streaming_geometry),
                );
            }
        }
        if cmp.training.window_start.is_empty() {
            ev.insert(
                "reason".to_string(),
                serde_json::json!("training path produced zero windows for this audio"),
            );
            LocalizationBucket::EmbeddingsDiffer
        } else if cmp.first_mismatch_idx == Some(0) || cmp.hash_match_frac < 0.5 {
            ev.insert(
                "reason".to_string(),
                serde_json::json!(
                    "first ordinal window hash mismatch OR <50% of ordinal windows match the training path"
                ),
            );
            LocalizationBucket::EmbeddingsDiffer
        } else {
            let mean_score = if pv.per_frame_scores.is_empty() {
                0.0
            } else {
                pv.per_frame_scores.iter().map(|s| s[0]).sum::<f32>()
                    / pv.per_frame_scores.len() as f32
            };
            ev.insert(
                "streaming_mean_score".to_string(),
                serde_json::json!(mean_score),
            );
            let boundary = training_p10.unwrap_or(NO_MATCH_RESET_THRESHOLD);
            ev.insert("training_p10".to_string(), serde_json::json!(boundary));
            ev.insert(
                "training_p10_source".to_string(),
                serde_json::json!(if training_p10.is_some() {
                    "training_in_sample_distribution"
                } else {
                    "fallback_no_match_reset_threshold"
                }),
            );
            if mean_score < boundary {
                // (c) embeddings match but scores are low.
                LocalizationBucket::EmbeddingsMatchScoresLow
            } else if pv.max_rolling_sum < MIN_CLASSIFIER_THRESHOLD {
                // (d) scores adequate but the rolling gate never accumulated.
                ev.insert(
                    "max_rolling_sum".to_string(),
                    serde_json::json!(pv.max_rolling_sum),
                );
                ev.insert(
                    "min_classifier_threshold".to_string(),
                    serde_json::json!(MIN_CLASSIFIER_THRESHOLD),
                );
                LocalizationBucket::ScoresAdequateGateNeverAccumulated
            } else {
                // (e) gate passed but the verifier stage blocked.
                ev.insert(
                    "max_rolling_sum".to_string(),
                    serde_json::json!(pv.max_rolling_sum),
                );
                ev.insert(
                    "verifier_score".to_string(),
                    serde_json::json!(pv.verifier_score),
                );
                ev.insert(
                    "verifier_threshold".to_string(),
                    serde_json::json!(pv.verifier_threshold),
                );
                if pv.agc_converged == Some(false) {
                    ev.insert(
                        "agc_note".to_string(),
                        serde_json::json!(
                            "agc_non_converged (correlated diagnostic, not causal — mahbot-1012)"
                        ),
                    );
                }
                LocalizationBucket::GatePassedVerifierRejected
            }
        }
    };
    (bucket, serde_json::Value::Object(ev))
}

/// Aggregate report for the mahbot-1012 same-audio comparison (Phase 8b).
#[derive(Default)]
struct Mahbot1012Report {
    phase_ms: f64,
    /// Dual-path comparison for every positive test variant.
    positive_comparisons: Vec<DualPathComparison>,
    /// Dual-path comparison for every training clip (control: same clip
    /// through both paths, raw original + cold pass).
    training_clip_comparisons: Vec<DualPathComparison>,
    /// Raw mel-frame comparison for one representative positive variant.
    mel_comparison: Option<MelFrameComparison>,
    /// Anchor probes for the representative subset (first variant per
    /// augmentation type).
    anchor_probes: Vec<AnchorProbeResult>,
    /// All training-side per-window classifier scores across positive
    /// variants (the in-sample distribution the (c)/(d) boundary uses).
    training_in_sample_scores: Vec<f32>,
    /// P10 of [`training_in_sample_scores`](Self::training_in_sample_scores).
    training_p10: Option<f32>,
    /// Exactly-one-bucket localization rows for every (variant, pass).
    localization_rows: Vec<LocalizationRow>,
}

impl Mahbot1012Report {
    fn bucket_counts(&self) -> std::collections::BTreeMap<&'static str, usize> {
        let mut counts: std::collections::BTreeMap<&'static str, usize> =
            std::collections::BTreeMap::new();
        for row in &self.localization_rows {
            *counts.entry(row.bucket.as_str()).or_insert(0) += 1;
        }
        counts
    }

    /// Per-variant localization row for a given pass, if present.
    fn row_for(&self, variant: &str, pass: &'static str) -> Option<&LocalizationRow> {
        self.localization_rows
            .iter()
            .find(|r| r.variant == variant && r.pass == pass)
    }

    fn to_json(&self) -> serde_json::Value {
        let mut localization_by_variant = serde_json::Map::new();
        for row in &self.localization_rows {
            let entry = localization_by_variant
                .entry(row.variant.clone())
                .or_insert_with(|| serde_json::json!({}));
            if let serde_json::Value::Object(map) = entry {
                map.insert(
                    row.pass.to_string(),
                    serde_json::json!({
                        "bucket": row.bucket.as_str(),
                        "evidence": row.evidence,
                    }),
                );
            }
        }
        serde_json::json!({
            "note": "Measurement-only (mahbot-1012): no production behavior or \
                     training-path changes.  All positive variants run the SAME PCM \
                     through both the enrollment/training embedding path and the \
                     streaming detection path (cold pass — fresh AGC, no warm-up).  \
                     The per-call mel dynamic-range floor hypothesis predicts \
                     window-0 hash mismatches across variants; the mel_frame_comparison \
                     settles the normalization-scope question (per-call global max vs \
                     frequency-axis) with raw frames.  INTERPRETATION CAVEAT: a \
                     window-0 hash mismatch can be caused by the mel-scope divergence \
                     OR by the padded-window geometry — the training side pads the \
                     full-utterance mel buffer while the streaming side pads the \
                     partial buffer accumulated at the first scoring step.  The \
                     per-window real_frames fields in dual_path_same_audio and the \
                     anchor_probe section separate the two; first_mismatch_idx loses \
                     discriminating power when every paired window mismatches (the \
                     observed case), so bucket (b) rows should be read together with \
                     real_frames and the anchor probe, not in isolation.",
            "phase_8b_same_audio_ms": self.phase_ms,
            "training_in_sample": {
                "n_per_window_scores": self.training_in_sample_scores.len(),
                "p10": self.training_p10,
                "deciles": deciles(&self.training_in_sample_scores),
                "note": "Training/enrollment-path per-window classifier scores \
                         applied to the SAME held-out test audio (VAD-gated, \
                         whole-utterance mel).  This is the apples-to-apples \
                         training-path score on each test variant, NOT the \
                         memorized in-sample training-window distribution.  \
                         Bucket (c) vs (d) boundary: streaming mean below P10 \
                         = low.",
            },
            "dual_path_same_audio": {
                "positive_variants": self
                    .positive_comparisons
                    .iter()
                    .map(DualPathComparison::to_json)
                    .collect::<Vec<_>>(),
                "training_clips_control": self
                    .training_clip_comparisons
                    .iter()
                    .map(DualPathComparison::to_json)
                    .collect::<Vec<_>>(),
            },
            "mel_frame_comparison": self.mel_comparison.as_ref().map(MelFrameComparison::to_json),
            "anchor_probe": self
                .anchor_probes
                .iter()
                .map(AnchorProbeResult::to_json)
                .collect::<Vec<_>>(),
            "localization": {
                "bucket_counts": self.bucket_counts(),
                "per_variant": localization_by_variant,
                "note": "Exactly-one-of-five buckets per (variant, pass) with the \
                         precedence (a) > (b) > (c) > (d) > (e).  AGC \
                         non-convergence is a correlated diagnostic attached as \
                         evidence, never a bucket.  Verifier training is \
                         entropy-seeded (None) — single-run bucket counts carry \
                         run-to-run variance; the hash/cosine/L2 evidence is \
                         seed-independent (classifier seed is fixed Some(0)).",
            },
        })
    }
}

/// Run the full mahbot-1012 same-audio comparison (Phase 8b).
///
/// Order of operations:
/// 1. Dual-path capture for every positive test variant.
/// 2. Dual-path capture for every training clip (raw original + cold pass —
///    the minimal reading that bounds runtime).
/// 3. Raw mel comparison for the first positive variant.
/// 4. Anchor probes for the representative subset: the first variant of each
///    of the 5 augmentation types (at most 5 probes).
/// 5. Training in-sample P10 from all positive dual-path training sides.
/// 6. Exactly-one-bucket localization for every (variant, pass) using the
///    warm/cold per-variant results from Phase 8.
fn run_mahbot1012_comparison(
    pos_variants: &[(Vec<f32>, String)],
    train_clips: &[(Vec<f32>, String)],
    classifier: &WakeWordClassifier,
    verifier: &VoiceVerifier,
    warm_pvs: &[PerVariantResult],
    cold_pvs: &[PerVariantResult],
) -> Mahbot1012Report {
    let start = Instant::now();
    let mut report = Mahbot1012Report::default();

    // ── 1. Positive variants: dual-path same-audio capture ────────────────
    eprintln!(
        "  mahbot-1012: dual-path same-audio capture ({} positive variants)...",
        pos_variants.len()
    );
    for (i, (pcm, label)) in pos_variants.iter().enumerate() {
        eprintln!(
            "    [{}/{}] {label} — streaming + training path...",
            i + 1,
            pos_variants.len()
        );
        let cmp = run_dual_path_capture(pcm, label, classifier, verifier);
        report
            .training_in_sample_scores
            .extend(cmp.training.classifier_scores.iter().copied());
        report.positive_comparisons.push(cmp);
    }

    // ── 2. Training clips through the streaming path (control) ────────────
    eprintln!(
        "  mahbot-1012: training clips through the streaming path ({} clips, raw original + cold pass)...",
        train_clips.len()
    );
    for (i, (pcm, label)) in train_clips.iter().enumerate() {
        eprintln!(
            "    [{}/{}] {label} — streaming + training path...",
            i + 1,
            train_clips.len()
        );
        let cmp = run_dual_path_capture(pcm, label, classifier, verifier);
        report.training_clip_comparisons.push(cmp);
    }

    // ── 3. Raw mel comparison for one representative variant ──────────────
    // Deterministic selection: the first positive variant (label order) that
    // produces both-side mel frames.  Documented so the report is reproducible.
    if let Some((pcm, label)) = pos_variants.first() {
        eprintln!("  mahbot-1012: raw mel frame comparison for representative variant {label}...");
        report.mel_comparison = run_mel_frame_comparison(pcm, label);
        if report.mel_comparison.is_none() {
            eprintln!("    (mel comparison skipped — no both-side mel frames)");
        }
    }

    // ── 4. Anchor probes: first variant per augmentation type ─────────────
    let mut seen_types: Vec<&'static str> = Vec::new();
    eprintln!(
        "  mahbot-1012: anchor-comparison probe (training grid vs natural streaming grid)..."
    );
    for (pcm, label) in pos_variants {
        let aug = augmentation_type(label);
        if seen_types.contains(&aug) {
            continue;
        }
        seen_types.push(aug);
        eprintln!("    {label} (augmentation {aug})...");
        if let Some(probe) = run_anchor_probe(pcm, label, classifier, verifier) {
            report.anchor_probes.push(probe);
        }
    }

    // ── 5. Training in-sample P10 ─────────────────────────────────────────
    report.training_p10 = deciles(&report.training_in_sample_scores).map(|d| d[1]);

    // ── 6. Localization buckets for every (variant, pass) ─────────────────
    let comparisons_by_label: std::collections::HashMap<&str, &DualPathComparison> = report
        .positive_comparisons
        .iter()
        .map(|c| (c.label.as_str(), c))
        .collect();
    let mut localize = |pvs: &[PerVariantResult], pass: &'static str| {
        for pv in pvs {
            // Unwrapping is safe by construction: positive_comparisons is keyed
            // by the same pos_variants labels that produce warm/cold
            // per-variant results, so every localized variant has a comparison
            // (the old Option fallback was dead code — mahbot-1012 reviewer).
            let cmp = comparisons_by_label
                .get(pv.variant.as_str())
                .copied()
                .unwrap_or_else(|| {
                    panic!(
                        "mahbot-1012: no dual-path comparison for positive variant {}",
                        pv.variant
                    )
                });
            let (bucket, evidence) = classify_localization_bucket(pv, cmp, report.training_p10);
            report.localization_rows.push(LocalizationRow {
                variant: pv.variant.clone(),
                pass,
                bucket,
                evidence,
            });
        }
    };
    localize(warm_pvs, "warm");
    localize(cold_pvs, "cold");

    report.phase_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "  mahbot-1012: comparison complete in {:.0}ms ({} positive + {} training-clip dual-path, {} anchor probes)",
        report.phase_ms,
        report.positive_comparisons.len(),
        report.training_clip_comparisons.len(),
        report.anchor_probes.len(),
    );
    report
}

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
                .map(|pv| pv_to_json(pv, Some("noise_overlap"), None))
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
// mahbot-1022: B-sweep measurement (measurement-only — no hot-path changes)
// ═══════════════════════════════════════════════════════════════════════════
//
// Measures the static-gate (post-fix) condition: how the Conv1D classifier
// score, the rolling sum, and the ring-4 verifier peak behave as a function
// of the streaming-layout mel BUFFER length B and the in-buffer scoring
// position.  The sweep scores the F1 enrollment clip (and its raw-level
// variants) through `score_single_embedding` with SEQUENTIAL ring semantics:
// positions 0 → 8 → 16 → 24 within one B-buffer share ONE embedding ring.
//
// Hard constraints honored here:
//   - No hot-path changes (`voice.rs` detection logic / thresholds untouched).
//   - No `set_classifier_weights` / `set_verifier` calls — classifier and
//     verifier are passed explicitly to `score_single_embedding` (the sweep
//     must not disturb the shared voice state).

/// B-sweep grid (mahbot-1022): streaming-layout mel buffer lengths B.
/// mahbot-1023: extended to 72/76/80 so the decisive verifier families at the
/// actual flush-aligned live trigger buffer lengths (B≈72–80, position 24 at
/// 48–56 real frames — OUTSIDE the original ≤68 grid) are measured, not
/// extrapolated (manager pin 2).  The sweep extension is for interpretability,
/// not a hard gate: the ≥3-run acceptance protocol validates the live behavior.
const SWEEP_BS: [usize; 10] = [44, 49, 52, 57, 60, 65, 68, 72, 76, 80];
/// In-buffer scoring positions (sequential ring order, stride 8).
const SWEEP_POSITIONS: [usize; 4] = [0, 8, 16, 24];
/// Decisive (B, position) cell for the F1-variant verdict (mahbot-1022 item 3a).
const DECISIVE_B: usize = 68;
const DECISIVE_POS: usize = 24;
/// Expected F1 streaming classifier score at the sanity anchor (mahbot-1022 item 5).
const BSWEEP_SANITY_EXPECTED: f32 = 0.2651;
/// Sanity-anchor divergence tolerance (mahbot-1022 item 5).
const BSWEEP_SANITY_TOLERANCE: f32 = 0.05;
/// Window-granularity cross-check real-frame tolerance (mahbot-1022 item 1):
/// the flush-duplicate range per window (0–3 frames per flush boundary).
const BSWEEP_CROSS_CHECK_TOLERANCE: usize = 3;

/// One (B, position) cell in the B-sweep (mahbot-1022).
#[derive(Clone)]
struct BsweepCell {
    b: usize,
    position: usize,
    measurable: bool,
    classifier_score: Option<f32>,       // None = unmeasurable
    verifier_ring4: Option<f32>,         // None = not evaluated (ring < 4)
    verifier_windows: Vec<f32>,          // per-window stride-1 predictions at ring 4
    verifier_window_rf: Vec<Vec<usize>>, // real-frame family per verifier window
    max_rolling_sum: f32,
    true_unique_bit_exact: usize,
    true_unique_tolerance: usize,
    geometry: &'static str,
}

impl BsweepCell {
    fn unmeasurable(b: usize, position: usize, geometry: &'static str) -> Self {
        Self {
            b,
            position,
            measurable: false,
            classifier_score: None,
            verifier_ring4: None,
            verifier_windows: Vec::new(),
            verifier_window_rf: Vec::new(),
            max_rolling_sum: 0.0,
            true_unique_bit_exact: 0,
            true_unique_tolerance: 0,
            geometry,
        }
    }
}

/// Ring-4 verifier detail at the decisive cell.  Outer `Option`: `None` = the
/// decisive cell itself was not measurable.  Inner `Option`: `None` = the cell
/// was measurable but the ring had not reached length 4 (verifier NOT
/// evaluated — never a zero-confidence 0.0).  `Some((max, per_window_scores,
/// per_window_real_frame_families))` = the ring-4 max over the two stride-1
/// windows.
type BsweepRing4Detail = Option<(Option<f32>, Vec<f32>, Vec<Vec<usize>>)>;

/// One raw-level F1 variant sweep result (mahbot-1022 item 3a).
struct BsweepVariantRaw {
    variant: String,
    mel_len: usize,
    confirm: Option<bool>, // None = decisive cell unmeasurable (mel < 68)
    b68_pos24_classifier: Option<f32>,
    b68_ring4_verifier_peak: Option<f32>,
    b68_max_rolling_sum: f32,
    /// Per-B grid (parallel to [`SWEEP_BS`]), cells in position order — the
    /// "sensitivity" of the classifier/verifier to the buffer length.
    grid: Vec<Vec<BsweepCell>>,
}

/// One F1-variant training-path result (mahbot-1022 item 3b).
struct BsweepVariantTraining {
    variant: String,
    produced: bool,
    n_windows: usize,
    per_window_classifier: Vec<f32>,
    per_window_verifier: Vec<f32>,
    max_rolling_sum: f32,
}

/// Aggregate B-sweep report (mahbot-1022).  `to_json(negatives)` emits the
/// additive `bsweep` JSON section merged with the same-run negatives extract.
struct BsweepReport {
    phase_ms: f64,
    f1_mel_len_shared: usize,
    f1_mel_len_fresh: usize,
    decisive_cell_measurable_shared: bool,
    decisive_cell_measurable_fresh: bool,
    dedup_epsilon: f32,
    dedup_epsilon_rule: &'static str,
    dedup_dist_min: f32,
    dedup_dist_median: f32,
    dedup_dist_max: f32,
    /// Per-B grid rows (parallel to [`SWEEP_BS`]); each row = cells in
    /// position order (sequential ring semantics within a B-buffer).
    grid: Vec<Vec<BsweepCell>>,
    /// Fresh-path per-B grid rows (dual-report when the VAD path changes
    /// measurability of the decisive cell — mahbot-1022 item 1).
    fresh_grid: Vec<Vec<BsweepCell>>,
    /// (B=68, position 24) ring-4 verifier detail.  Outer `None` = decisive
    /// cell not measurable; inner `Option<f32>` = ring-4 max (`None` = ring <
    /// 4, NOT evaluated).
    decisive_ring4: BsweepRing4Detail,
    variants_raw: Vec<BsweepVariantRaw>,
    variants_training: Vec<BsweepVariantTraining>,
    confirm_count: usize,
    measurable_count: usize,
    passed_4of5: bool,
    unmeasurable_variants: Vec<String>,
    sanity_measured: Option<f32>,
    classifier_fingerprint: String,
    verifier_fingerprint: String,
    verifier_runtime_threshold: f32,
    cross_check: serde_json::Value,
    trailing_post_trim: serde_json::Value,
}

/// Streaming-layout mel for a PCM clip (mahbot-1022): AGC via a fresh
/// [`AudioPreprocessor`], then VAD-gated streaming mel extraction with either
/// the SHARED live detector (faithful to the streaming pipeline) or a FRESH
/// detector (the enrollment/training path's per-clip fresh detector).
/// Returns `(mel_frames, mel_len)`.
fn bsweep_streaming_mel(pcm: &[f32], shared: bool) -> (Vec<Vec<f32>>, usize) {
    let agc = agc_pcm(pcm);
    let (mel, _speech_audio) = if shared {
        let detector =
            super::VAD_DETECTOR.get_or_init(|| std::sync::Mutex::new(earshot::Detector::default()));
        let mut detector = detector.lock().unwrap_poison();
        super::vad_gate_streaming_mel(&agc, |hop| {
            super::is_speech_with_detector(hop, &mut detector, super::VAD_THRESHOLD)
        })
    } else {
        let mut detector = Detector::default();
        super::vad_gate_streaming_mel(&agc, |hop| {
            super::is_speech_with_detector(hop, &mut detector, super::VAD_THRESHOLD)
        })
    };
    let mel_len = mel.len();
    (mel, mel_len)
}

/// Relative L2 distance between two equal-length mel frames:
/// `||a - b||_2 / max(||a||_2, ||b||_2, 1e-8)`.
fn frame_relative_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut diff_sq = 0.0_f32;
    let mut a_sq = 0.0_f32;
    let mut b_sq = 0.0_f32;
    for (x, y) in a.iter().zip(b) {
        let d = x - y;
        diff_sq += d * d;
        a_sq += x * x;
        b_sq += y * y;
    }
    diff_sq.sqrt() / (a_sq.sqrt().max(b_sq.sqrt()).max(1e-8))
}

/// Derive the dedup epsilon from the sorted boundary-frame relative-L2
/// distance distribution (mahbot-1022 item 4).
///
/// Sorts ascending and finds the largest multiplicative gap `d[i+1]/d[i]`
/// between adjacent sorted POSITIVE values.  If that max ratio is `>= 2.0`
/// the distribution is a documented bimodal split (duplicate frames clustered
/// near 0, distinct frames further out) and epsilon = the geometric mean
/// `sqrt(d[i] * d[i+1])` of the gap pair.  Otherwise epsilon = 0.0, meaning
/// the tolerance definition collapses to bit-exact.  Returns
/// `(epsilon, rule_description)`.
///
/// Exact-zero distances (bit-identical frames) sit at the FLOOR of the
/// duplicate cluster, so the 0 → first-nonzero boundary is NOT treated as a
/// gap candidate: treating it as infinite would always win AND would yield a
/// degenerate `sqrt(0 * next) = 0`, collapsing the tolerance to bit-exact
/// even when a clear bimodal split exists above the zeros.  Only ratios
/// between adjacent positive values can select the split.
fn derive_dedup_epsilon(distances: &mut [f32]) -> (f32, &'static str) {
    if distances.len() < 2 {
        return (0.0, "insufficient distances (< 2) — tolerance ≡ bit-exact");
    }
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut best_ratio = 0.0_f32;
    let mut best_idx = 0usize;
    for i in 0..distances.len() - 1 {
        // Exact-zero distances are the duplicate-cluster floor; a 0 → nonzero
        // boundary carries no split information (the ratio is undefined, and
        // its geometric mean would be degenerate 0).
        if distances[i] <= 0.0 {
            continue;
        }
        let ratio = distances[i + 1] / distances[i];
        if ratio > best_ratio {
            best_ratio = ratio;
            best_idx = i;
        }
    }
    if best_ratio >= 2.0 {
        let epsilon = (distances[best_idx] * distances[best_idx + 1]).sqrt();
        (
            epsilon,
            "bimodal split: largest positive multiplicative gap >= 2.0; \
             epsilon = geometric mean of the gap pair (exact-zero distances \
             form the duplicate-cluster floor and are not gap candidates)",
        )
    } else {
        (
            0.0,
            "no bimodal split (max positive gap ratio < 2.0) — tolerance ≡ bit-exact",
        )
    }
}

/// Count true-unique frames in the window-scoped mel slice `[position..b]`
/// (mahbot-1022 item 4): `1 + count of i in [position+1..b] where frame[i]`
/// is NOT a duplicate of `frame[i-1]`.
///
/// Returns `(bit_exact_count, tolerance_count)`:
/// - Bit-exact: a duplicate requires all 32 bands bit-identical
///   (`x.to_bits() == y.to_bits()` per band).
/// - Tolerance: a duplicate requires `frame_relative_l2 <= epsilon`; when
///   `epsilon == 0.0` this definition is identical to bit-exact.
fn count_true_unique(mel: &[Vec<f32>], position: usize, b: usize, epsilon: f32) -> (usize, usize) {
    if position >= b || position >= mel.len() {
        return (0, 0);
    }
    let end = b.min(mel.len());
    let mut bit_exact = 1usize;
    let mut tolerance = 1usize;
    for i in (position + 1)..end {
        let prev = &mel[i - 1];
        let cur = &mel[i];
        let same_bits = prev.len() == cur.len()
            && prev
                .iter()
                .zip(cur)
                .all(|(x, y)| x.to_bits() == y.to_bits());
        if !same_bits {
            bit_exact += 1;
        }
        if epsilon > 0.0 {
            if frame_relative_l2(prev, cur) > epsilon {
                tolerance += 1;
            }
        } else {
            tolerance = bit_exact;
        }
    }
    (bit_exact, tolerance)
}

/// Stride-1 verifier windows at ring length 4 exactly (mahbot-1022 item 2).
///
/// The two windows
/// [`score_single_embedding`](super::score_single_embedding) evaluates once
/// the ring reaches [`VERIFIER_WARMUP_EMBEDDINGS`] (ring start 0 and 1).
/// Returns `Some((max, per_window_scores, real_frame_families))`; `None`
/// when the ring length is not exactly 4 (verifier NOT evaluated — the
/// caller reports ring < 4 as JSON null, never as a zero-confidence 0.0).
///
/// The real-frame family for window start `i` is the real mel-frame count of
/// each embedding in that window (`ring[i..i+3]`), computed from the ACTUAL
/// ring embedding positions (`positions[i+j]`) as `b - positions[i+j]` — i.e.
/// the B, B-8, B-16 / B-8, B-16, B-24 families for the fixed 0/8/16/24 grid
/// (each family describes the window it is attached to, so the report's
/// per-window scores and families stay consistent).
fn ring4_window_scores(
    ring: &[Vec<f32>],
    verifier: &VoiceVerifier,
    b: usize,
    positions: &[usize],
) -> Option<(f32, Vec<f32>, Vec<Vec<usize>>)> {
    if ring.len() != 4 {
        return None;
    }
    let mut window = [0.0_f32; crate::VERIFIER_INPUT_DIM];
    let mut scores = Vec::with_capacity(2);
    let mut families = Vec::with_capacity(2);
    for i in 0..2 {
        crate::audio::voice_verifier::fill_verifier_window(ring, i, &mut window);
        scores.push(verifier.predict(&window));
        let family: Vec<usize> = (0..crate::VERIFIER_WINDOW_SIZE)
            .map(|j| {
                let pos = positions.get(i + j).copied().unwrap_or(0);
                b.saturating_sub(pos)
            })
            .collect();
        families.push(family);
    }
    let max = scores.iter().copied().fold(0.0_f32, f32::max);
    Some((max, scores, families))
}

/// SHA-256 fingerprint of classifier weights (mahbot-1022 item 6): sha2-256
/// over the f32 LE bytes of each field in documented order — conv1_weight,
/// conv1_bias, bn1_gamma, bn1_beta, bn1_running_mean, bn1_running_var,
/// conv2_weight, conv2_bias, bn2_gamma, bn2_beta, bn2_running_mean,
/// bn2_running_var, fc_weight, fc_bias — then `bn_eps` (f32 LE), then
/// `arch.conv1_out`, `arch.conv2_out`, `arch.kernel_size` (u32 LE each).
/// Returns [`crate::util::hex_string`].
fn weights_fingerprint_classifier(
    w: &crate::audio::wake_word_classifier::ClassifierWeights,
) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let mut push_f32s = |data: &[f32]| {
        for v in data {
            hasher.update(v.to_le_bytes());
        }
    };
    push_f32s(&w.conv1_weight);
    push_f32s(&w.conv1_bias);
    push_f32s(&w.bn1_gamma);
    push_f32s(&w.bn1_beta);
    push_f32s(&w.bn1_running_mean);
    push_f32s(&w.bn1_running_var);
    push_f32s(&w.conv2_weight);
    push_f32s(&w.conv2_bias);
    push_f32s(&w.bn2_gamma);
    push_f32s(&w.bn2_beta);
    push_f32s(&w.bn2_running_mean);
    push_f32s(&w.bn2_running_var);
    push_f32s(&w.fc_weight);
    push_f32s(&w.fc_bias);
    hasher.update(w.bn_eps.to_le_bytes());
    hasher.update((w.arch.conv1_out as u32).to_le_bytes());
    hasher.update((w.arch.conv2_out as u32).to_le_bytes());
    hasher.update((w.arch.kernel_size as u32).to_le_bytes());
    crate::util::hex_string(&hasher.finalize())
}

/// SHA-256 fingerprint of the verifier (mahbot-1022 item 6): sha2-256 over
/// the f32 LE bytes of `conv_weight`, `conv_bias`, `fc_weight`, `fc_bias`,
/// THEN the runtime-calibrated `threshold`.  Returns
/// [`crate::util::hex_string`].
fn weights_fingerprint_verifier(v: &VoiceVerifier) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    // Hash the primary member's weights, then every ensemble member's weights
    // in order (mahbot-1025) so the fingerprint distinguishes the full
    // multi-seed ensemble, not just member 0.
    let mut member_views: Vec<&VoiceVerifier> = vec![v];
    member_views.extend(v.ensemble_members.iter());
    for member in member_views {
        for data in [
            &member.conv_weight,
            &member.conv_bias,
            &member.fc_weight,
            &member.fc_bias,
        ] {
            for val in data {
                hasher.update(val.to_le_bytes());
            }
        }
    }
    hasher.update(v.threshold.to_le_bytes());
    crate::util::hex_string(&hasher.finalize())
}

/// Per-B-buffer scoring context (mahbot-1022): the sequential embedding ring,
/// the in-buffer position of each ring embedding, and the Conv1D score window
/// shared by ALL cells of ONE B-buffer (positions 0 → 8 → 16 → 24 share one
/// embedding ring — the pinned sequential-ring semantics).
struct BsweepCellContext {
    ring: Vec<Vec<f32>>,
    ring_positions: Vec<usize>,
    score_window: Vec<f32>,
}

impl BsweepCellContext {
    fn new() -> Self {
        Self {
            ring: Vec::new(),
            ring_positions: Vec::new(),
            score_window: Vec::new(),
        }
    }
}

/// Score one (B, position) cell with SEQUENTIAL ring semantics (mahbot-1022):
/// slice `mel[position..b]` (truncated to the 76-frame embedding window —
/// mahbot-1023 sweep extension), pad to the 76-frame embedding window, compute
/// one embedding, then `score_single_embedding` with a static gate (no
/// adaptive state, no candidate tracking).  `ctx` persists across cells within
/// a B-buffer so positions 0 → 8 → 16 → 24 share ONE embedding ring.  At ring
/// length 4 exactly the per-window verifier scores are captured via
/// [`ring4_window_scores`].
#[expect(clippy::too_many_arguments)]
fn bsweep_score_cell(
    models: &super::OnnxModels,
    mel: &[Vec<f32>],
    position: usize,
    b: usize,
    classifier: &WakeWordClassifier,
    verifier: &VoiceVerifier,
    ctx: &mut BsweepCellContext,
    epsilon: f32,
    geometry: &'static str,
) -> BsweepCell {
    // The embedding window always spans exactly EMBEDDING_WINDOW_FRAMES (76)
    // mel frames: at B > 76 the window at `position` truncates at
    // `position + 76` (the frames beyond the window are not part of the
    // scored window).  The real-frame counts reported via count_true_unique
    // use the same truncation so the true-unique band matches the window.
    let window_end = b.min(position + super::EMBEDDING_WINDOW_FRAMES);
    let slice = &mel[position..window_end];
    let padded = super::pad_mel_frames_to_window(slice);
    // A failed embedding (e.g. model unavailable) makes this cell
    // UNMEASURABLE and must NOT advance the shared ring — a stale embedding
    // would corrupt the sequential-ring scores of later cells in this B-buffer.
    let Ok(emb) = super::compute_embedding(models, &padded) else {
        return BsweepCell::unmeasurable(b, position, geometry);
    };
    let (_, rolling_sum, total_score, _, _, _) = super::score_single_embedding(
        &emb,
        &mut ctx.ring,
        Some(classifier),
        Some(verifier),
        &mut ctx.score_window,
        None, // static gate — no adaptive state (mahbot-1022)
        super::ADAPTIVE_K_DEFAULT,
        None, // no candidate tracking on the sweep
    );
    ctx.ring_positions.push(position);
    // Ring length must be EXACTLY 4 (the gate must match ring4_window_scores'
    // contract): a hypothetical ring > 4 would otherwise report a silent
    // Some(0.0) with empty windows.  On the fixed 0/8/16/24 grid the ring
    // reaches 4 exactly at position 24.
    let (verifier_ring4, verifier_windows, verifier_window_rf) =
        match ring4_window_scores(&ctx.ring, verifier, b, &ctx.ring_positions) {
            Some((max, wins, fams)) => (Some(max), wins, fams),
            None => (None, Vec::new(), Vec::new()),
        };
    let (true_uniq_bit, true_uniq_tol) = count_true_unique(mel, position, window_end, epsilon);
    BsweepCell {
        b,
        position,
        measurable: true,
        classifier_score: Some(total_score),
        verifier_ring4,
        verifier_windows,
        verifier_window_rf,
        max_rolling_sum: rolling_sum,
        true_unique_bit_exact: true_uniq_bit,
        true_unique_tolerance: true_uniq_tol,
        geometry,
    }
}

/// Score the full B × position grid over one streaming-layout mel
/// (mahbot-1022).  Each B-buffer uses a fresh ring / score window; positions
/// within a B-buffer share one ring (sequential semantics).  Returns
/// `(grid, decisive_ring4)` where `decisive_ring4` is the ring-4 verifier
/// detail at (B=68, position 24): outer `None` = decisive cell not measurable;
/// inner `Option<f32>` propagates the cell's ring-4 value (`None` = ring < 4,
/// NOT evaluated).
fn bsweep_score_grid(
    mel: &[Vec<f32>],
    classifier: &WakeWordClassifier,
    verifier: &VoiceVerifier,
    epsilon: f32,
    geometry: &'static str,
) -> (Vec<Vec<BsweepCell>>, BsweepRing4Detail) {
    let models = super::ONNX_MODELS
        .get()
        .expect("ONNX models must be loaded by the benchmark");
    let mut grid = Vec::with_capacity(SWEEP_BS.len());
    let mut decisive_ring4 = None;
    for &b in &SWEEP_BS {
        // Every B-buffer is the FIRST B streaming-layout frames from speech
        // onset (flush-boundary duplicates included) — the pinned geometry
        // (the caller passes the grid's path label: shared vs fresh VAD).
        let mut ctx = BsweepCellContext::new();
        let mut cells = Vec::with_capacity(SWEEP_POSITIONS.len());
        for &pos in &SWEEP_POSITIONS {
            if b > mel.len() || pos >= b {
                // B exceeds the streaming-layout mel length, or the position is
                // outside the [0..B) window → unmeasurable.
                cells.push(BsweepCell::unmeasurable(b, pos, geometry));
                continue;
            }
            let cell = bsweep_score_cell(
                models, mel, pos, b, classifier, verifier, &mut ctx, epsilon, geometry,
            );
            if b == DECISIVE_B && pos == DECISIVE_POS {
                // Propagate the cell's verifier Option: a measurable cell with
                // ring < 4 (mid-grid embedding failure) stays NOT-evaluated
                // (JSON null), never a plausible-looking 0.0.
                decisive_ring4 = Some((
                    cell.verifier_ring4,
                    cell.verifier_windows.clone(),
                    cell.verifier_window_rf.clone(),
                ));
            }
            cells.push(cell);
        }
        grid.push(cells);
    }
    (grid, decisive_ring4)
}

/// Run the F1 training-path (enrollment-side) variant scoring (mahbot-1022
/// item 3b).  The training path is "post-AGC/VAD": AGC the raw F1 PCM, gate
/// it with a FRESH detector, then augment the VAD-gated speech into the five
/// variants (original / speed-down 0.95x / speed-up 1.05x conditional on the
/// VAD-gated speech pre-pad ≥ 500 ms / vol-down −3 dB / pink noise SNR 25 dB
/// seed 0) — mirroring [`vad_segment_and_enroll`]'s per-utterance augmentation.
/// Each produced variant is embedded via the whole-utterance mel + stride-8
/// embeddings path and scored full-sequence per window.  Returns one
/// [`BsweepVariantTraining`] per produced variant (empty when no speech).
fn bsweep_training_path_scores(
    f1_pcm: &[f32],
    classifier: &WakeWordClassifier,
    verifier: &VoiceVerifier,
) -> Vec<BsweepVariantTraining> {
    let models = super::ONNX_MODELS
        .get()
        .expect("ONNX models must be loaded by the benchmark");
    let agc = agc_pcm(f1_pcm);
    if agc.is_empty() {
        return Vec::new();
    }
    let mut detector = Detector::default();
    let (_streaming_mel, speech_audio) = super::vad_gate_streaming_mel(&agc, |hop| {
        super::is_speech_with_detector(hop, &mut detector, super::VAD_THRESHOLD)
    });
    if speech_audio.is_empty() {
        return Vec::new();
    }

    // ── Augment the VAD-gated speech (vad_segment_and_enroll semantics) ──
    // utterance index 0 (single F1 clip) → noise seed 0.
    let utterance_idx = 0usize;
    let speed_down =
        crate::audio::tts_data_gen::speed_perturbation(&speech_audio, TARGET_SAMPLE_RATE, 0.95);
    let pre_pad_duration_samples = speech_audio
        .len()
        .saturating_sub(2 * super::CONTEXT_PADDING_SAMPLES);
    let pre_pad_duration_ms =
        (pre_pad_duration_samples as u64 * 1000) / u64::from(TARGET_SAMPLE_RATE);
    let speed_up = if pre_pad_duration_ms >= 500 {
        Some(crate::audio::tts_data_gen::speed_perturbation(
            &speech_audio,
            TARGET_SAMPLE_RATE,
            1.05,
        ))
    } else {
        None
    };
    let volume_down = crate::util::apply_gain(&speech_audio, -3.0);
    let noise = crate::util::add_noise(&speech_audio, 25.0, utterance_idx as u64);

    // Variants in push order: original, speed-down, vol-down, noise, then
    // speed-up chained LAST (conditional on VAD-gated pre-pad >= 500 ms).
    // (The speed-up gate differs from the raw-level TTS pre-pad gate of
    // item 3a — the produced-variant counts are reported per semantics.)
    let variants: Vec<(String, Vec<f32>)> = [
        ("original".to_string(), speech_audio.clone()),
        ("speed_down".to_string(), speed_down),
        ("vol_down".to_string(), volume_down),
        ("noise".to_string(), noise),
    ]
    .into_iter()
    .chain(speed_up.map(|sp| ("speed_up".to_string(), sp)))
    .collect();

    let mut results = Vec::with_capacity(variants.len());
    for (variant_label, variant_pcm) in variants {
        match super::compute_mel_spectrogram(models, &variant_pcm) {
            Ok(mel_frames) => match super::embeddings_from_mel_frames(models, &mel_frames) {
                Ok(embeddings) if !embeddings.is_empty() => {
                    let (scores, v_scores, max_rs) =
                        score_embedding_sequence_with(&embeddings, classifier, verifier);
                    results.push(BsweepVariantTraining {
                        variant: variant_label,
                        produced: true,
                        n_windows: scores.len(),
                        per_window_classifier: scores,
                        per_window_verifier: v_scores,
                        max_rolling_sum: max_rs,
                    });
                }
                _ => results.push(BsweepVariantTraining {
                    variant: variant_label,
                    produced: false,
                    n_windows: 0,
                    per_window_classifier: Vec::new(),
                    per_window_verifier: Vec::new(),
                    max_rolling_sum: 0.0,
                }),
            },
            Err(_) => results.push(BsweepVariantTraining {
                variant: variant_label,
                produced: false,
                n_windows: 0,
                per_window_classifier: Vec::new(),
                per_window_verifier: Vec::new(),
                max_rolling_sum: 0.0,
            }),
        }
    }
    results
}

/// Same-run F1 streaming classifier score at the sanity anchor (mahbot-1022
/// item 5): the F1 training-clip comparison's streaming-side window with
/// start 3, else real_frames == 76, else the largest-real-frame window.
fn bsweep_sanity_anchor_score(same_audio: &Mahbot1012Report) -> Option<f32> {
    let cmp = same_audio
        .training_clip_comparisons
        .iter()
        .find(|c| c.label == "F1.json_enroll0")?;
    let streaming = &cmp.streaming;
    let idx = streaming
        .window_start
        .iter()
        .position(|&s| s == 3)
        .or_else(|| {
            streaming
                .real_frames
                .iter()
                .position(|&r| r == super::EMBEDDING_WINDOW_FRAMES)
        })
        .or_else(|| {
            streaming
                .real_frames
                .iter()
                .enumerate()
                .max_by_key(|&(_, r)| r)
                .map(|(i, _)| i)
        })?;
    streaming.classifier_scores.get(idx).copied()
}

/// Window-granularity consistency check (mahbot-1022 item 1): cross-check the
/// sweep's recomputed shared-path mel window structure against the same-run F1
/// streaming trajectory's recorded windows.
///
/// Pass criterion: for every recorded FULL-LENGTH window (recorded geometry
/// `true_sliding`, i.e. the window spans the full 76-frame embedding window),
/// expected real frames = `min(shared_mel_len - start, 76)` and agree =
/// `|recorded - expected| <= 3` (the flush-duplicate range).  Pass = all
/// comparable windows agree AND at least one comparable window exists.
///
/// Short-buffer / trimmed-buffer windows (`padded_fallback`, `cold_start_tiled`
/// — recorded real frames < 76) are INFORMATIONAL: their recorded real-frame
/// count reflects the streaming buffer state AT SCORING TIME (the deferred
/// burst / segment-end pass padded geometry, or a post-trim re-score with
/// buffer-relative starts), which the sweep's one-shot recomputed full mel
/// cannot reproduce.
/// They never force `pass = false` — that is the legitimate run-position VAD
/// divergence the criterion tolerates (the sweep recomputes the shared mel
/// AFTER Phase 8b, advancing the shared detector).
fn bsweep_cross_check(same_audio: &Mahbot1012Report, shared_mel_len: usize) -> serde_json::Value {
    let f1_cmp = same_audio
        .training_clip_comparisons
        .iter()
        .find(|c| c.label == "F1.json_enroll0");
    let mut windows = Vec::new();
    let mut n_comparable = 0usize;
    let mut n_agreeing = 0usize;
    let mut all_agree = true;
    if let Some(cmp) = f1_cmp {
        let streaming = &cmp.streaming;
        for i in 0..streaming.window_start.len() {
            let recorded_start = streaming.window_start[i];
            let recorded_rf = streaming.real_frames.get(i).copied().unwrap_or(0);
            let recorded_geometry = streaming.geometry.get(i).copied().unwrap_or("");
            // Full-length window = the full 76-frame embedding window
            // (true_sliding geometry; the rf == 76 fallback covers a missing
            // geometry label).  Everything else is a short/trimmed-buffer
            // window the recomputed full mel cannot verify.
            let comparable = recorded_geometry == "true_sliding"
                || recorded_rf >= super::EMBEDDING_WINDOW_FRAMES;
            let expected_rf = shared_mel_len
                .saturating_sub(recorded_start)
                .min(super::EMBEDDING_WINDOW_FRAMES);
            let delta = recorded_rf.abs_diff(expected_rf);
            let agree = comparable && delta <= BSWEEP_CROSS_CHECK_TOLERANCE;
            if comparable {
                n_comparable += 1;
                if delta <= BSWEEP_CROSS_CHECK_TOLERANCE {
                    n_agreeing += 1;
                } else {
                    all_agree = false;
                }
            }
            windows.push(serde_json::json!({
                "recorded_start": recorded_start,
                "recorded_real_frames": recorded_rf,
                "recorded_geometry": recorded_geometry,
                "expected_real_frames": expected_rf,
                "delta_frames": delta,
                "comparable": comparable,
                "agree": agree,
                "note": if comparable {
                    serde_json::Value::Null
                } else {
                    serde_json::json!(
                        "short-buffer/trimmed-buffer window — recorded rf reflects \
                         the buffer state at scoring time (burst/pass padded \
                         geometry or post-trim buffer-relative starts), which the \
                         recomputed full mel cannot reproduce; informational only"
                    )
                },
            }));
        }
    }
    let pass = n_comparable > 0 && all_agree;
    serde_json::json!({
        "method": "window-granularity real-frame agreement vs the same-run F1 \
                   streaming trajectory",
        "criterion": "per recorded FULL-LENGTH (true_sliding) window: expected real \
                      frames = min(shared_mel_len - start, 76); agree = \
                      |recorded - expected| <= 3 (flush-duplicate range).  Pass = \
                      all comparable windows agree (and >= 1 comparable).  \
                      Short-buffer / trimmed-buffer windows are informational.",
        "shared_mel_len": shared_mel_len,
        "n_recorded_windows": windows.len(),
        "n_comparable_windows": n_comparable,
        "n_agreeing_windows": n_agreeing,
        "pass": pass,
        "windows": windows,
        "note": "Same-run trajectory (Phase 8b) vs the sweep's post-Phase-8b \
                 recomputed shared-path mel.  Run-position-dependent shared VAD \
                 state can legitimately shift the recomputed mel length; a \
                 divergent length shows up as |delta| > 3 on the full-length \
                 windows and flags pass=false with the per-window table for \
                 inspection.",
    })
}

/// Trailing post-trim family (mahbot-1022 item 13): the same-run F1
/// streaming trajectory's maximal suffix with `real_frames >= 50` — the
/// window-start set is READ FROM THE RUN, not assumed.
fn bsweep_trailing_post_trim(same_audio: &Mahbot1012Report) -> serde_json::Value {
    let f1_cmp = same_audio
        .training_clip_comparisons
        .iter()
        .find(|c| c.label == "F1.json_enroll0");
    let mut windows = Vec::new();
    if let Some(cmp) = f1_cmp {
        let streaming = &cmp.streaming;
        let mut start_idx = streaming.real_frames.len();
        while start_idx > 0 && streaming.real_frames[start_idx - 1] >= 50 {
            start_idx -= 1;
        }
        for i in start_idx..streaming.real_frames.len() {
            windows.push(serde_json::json!({
                "window_start": streaming.window_start.get(i),
                "real_frames": streaming.real_frames.get(i),
                "classifier_score": streaming.classifier_scores.get(i),
                "verifier_score": streaming.verifier_scores.get(i),
            }));
        }
    }
    serde_json::json!({
        "geometry": "trailing-post-trim",
        "windows": windows,
        "note": "Maximal suffix of the same-run F1 streaming trajectory with \
                 real_frames >= 50.  Window-start set is READ FROM THE RUN, not assumed.",
    })
}

/// Verdict helper (mahbot-1022 item 3a): `(confirm_count, measurable_count,
/// passed)` from per-variant confirms where `None` = unmeasurable (mel < 68).
/// 4-of-5 over the measurable variants: at least 4 measurable AND at least 4
/// confirmed (fewer than 4 measurable variants cannot reach 4 confirms).
fn bsweep_verdict_passes(confirms: &[Option<bool>]) -> (usize, usize, bool) {
    let confirm_count = confirms.iter().filter(|c| c.is_some_and(|v| v)).count();
    let measurable_count = confirms.iter().filter(|c| c.is_some()).count();
    let passed = confirm_count >= 4 && measurable_count >= 4;
    (confirm_count, measurable_count, passed)
}

/// Same-run negatives verifier extraction (mahbot-1022 item 4): gate crossing
/// = any frame with `per_frame_scores[i][ROLLING_SUM_IDX] >=
/// MIN_CLASSIFIER_THRESHOLD`.  Reports total negatives, gate-crossing count,
/// and per crossing the first-crossing-frame verifier value +
/// per-variant peak verifier value.  Labeled same-run.
fn negative_verifier_extraction(all_neg_pv: &[(&PerVariantResult, String)]) -> serde_json::Value {
    let mut crossings = Vec::new();
    let mut per_variant_peak = Vec::new();
    let mut gate_crossing_count = 0usize;
    for (pv, category) in all_neg_pv {
        let first_crossing_idx = pv
            .per_frame_scores
            .iter()
            .position(|frame| frame[ROLLING_SUM_IDX] >= MIN_CLASSIFIER_THRESHOLD);
        if let Some(idx) = first_crossing_idx {
            gate_crossing_count += 1;
            let first_crossing_verifier = pv
                .verifier_score_trajectory
                .get(idx)
                .copied()
                .unwrap_or(0.0);
            crossings.push(serde_json::json!({
                "variant": pv.variant,
                "category": category,
                "first_crossing_frame_idx": idx,
                "first_crossing_verifier_value": first_crossing_verifier,
                "peak_verifier_value": pv.verifier_score,
            }));
        }
        per_variant_peak.push(serde_json::json!({
            "variant": pv.variant,
            "category": category,
            "peak_verifier_value": pv.verifier_score,
        }));
    }
    serde_json::json!({
        "note": "Same-run negatives (Phase 9-14): gate crossing = any frame with \
                 rolling_sum >= MIN_CLASSIFIER_THRESHOLD (2.13).  Prior run: 9/59.",
        "total_negatives": all_neg_pv.len(),
        "gate_crossing_count": gate_crossing_count,
        "per_crossing": crossings,
        "per_variant_peak_verifier": per_variant_peak,
    })
}

fn bsweep_cell_to_json(cell: &BsweepCell) -> serde_json::Value {
    serde_json::json!({
        "b": cell.b,
        "position": cell.position,
        "measurable": cell.measurable,
        "classifier_score": cell.classifier_score,
        "verifier_ring4": cell.verifier_ring4,
        "verifier_windows": cell.verifier_windows,
        "verifier_window_rf": cell.verifier_window_rf,
        // Raw buffer length == B (the slice [position..b] is taken from the
        // exactly-B streaming-layout buffer); emitted explicitly so every cell
        // row carries the ticket's raw-buffer-length field without a separate
        // drifiable struct field.
        "raw_buffer_len": cell.b,
        "true_unique_bit_exact": cell.true_unique_bit_exact,
        "true_unique_tolerance": cell.true_unique_tolerance,
        // Explicit divergence flag (mahbot-1022 item 1): when the two dedup
        // definitions disagree on a confirm() outcome's true-unique count,
        // the divergence is flagged here rather than silently choosing one.
        "dedup_definitions_disagree": cell.measurable
            && cell.true_unique_bit_exact != cell.true_unique_tolerance,
        "max_rolling_sum": cell.max_rolling_sum,
        "geometry": cell.geometry,
    })
}

/// Run the full B-sweep measurement (mahbot-1022).  Returns `None` when the
/// F1 training clip (`"F1.json_enroll0"`) is absent from `train_clips`.
#[expect(clippy::too_many_lines)]
fn run_bsweep(
    train_clips: &[(Vec<f32>, String)],
    classifier: &WakeWordClassifier,
    verifier: &VoiceVerifier,
    same_audio: &Mahbot1012Report,
) -> Option<BsweepReport> {
    let start = Instant::now();

    // ── 1. Locate F1 by label ─────────────────────────────────────────────
    let (f1_pcm, _f1_label) = train_clips.iter().find(|(_, l)| l == "F1.json_enroll0")?;

    // ── 2. Mel length validation on BOTH paths (early) ────────────────────
    // Primary path = SHARED detector (faithful to the live pipeline);
    // dual-report FRESH path because measurability differs (analysis:
    // fresh ~57-58 < 68, shared >= 79).
    let (shared_mel, shared_len) = bsweep_streaming_mel(f1_pcm, true);
    let (fresh_mel, fresh_len) = bsweep_streaming_mel(f1_pcm, false);
    eprintln!(
        "  B-sweep: F1 streaming-layout mel — shared detector: {shared_len} frames, fresh detector: {fresh_len} frames"
    );
    let decisive_cell_measurable_shared = DECISIVE_B <= shared_len;
    let decisive_cell_measurable_fresh = DECISIVE_B <= fresh_len;

    // ── 3. Dedup: boundary-frame difference distribution (shared mel) ─────
    let mut boundary_dists: Vec<f32> = shared_mel
        .windows(2)
        .map(|w| frame_relative_l2(&w[0], &w[1]))
        .collect();
    let (dedup_epsilon, dedup_epsilon_rule) = derive_dedup_epsilon(&mut boundary_dists);
    let dist_min = boundary_dists.first().copied().unwrap_or(0.0);
    let dist_median = boundary_dists
        .get(boundary_dists.len() / 2)
        .copied()
        .unwrap_or(0.0);
    let dist_max = boundary_dists.last().copied().unwrap_or(0.0);

    // ── 4. Primary F1 grid (shared path) ──────────────────────────────────
    let (grid, decisive_ring4) = bsweep_score_grid(
        &shared_mel,
        classifier,
        verifier,
        dedup_epsilon,
        "streaming-layout-first-B",
    );
    // Dual-report the FRESH-path grid when the VAD path changes whether the
    // decisive (B=68, position 24) cell is measurable.  The verdict uses the
    // shared path; the fresh grid is a labeled comparison (cells with B >
    // fresh_len are UNMEASURABLE).
    let (fresh_grid, _fresh_ring4) = bsweep_score_grid(
        &fresh_mel,
        classifier,
        verifier,
        dedup_epsilon,
        "streaming-layout-first-B-fresh-vad",
    );

    // ── 5. F1 raw-level variants (item 3a) ────────────────────────────────
    // `pcm_augment_enrollment_variants` produces original / speed-down /
    // speed-up (conditional, >= 500 ms) / vol-down / pink-noise (SNR 25 dB,
    // seed = clip index 0) at the RAW level — the same PCM-level augment set
    // the detection test uses (mahbot-932 Fix 5 semantics).
    let f1_variants =
        pcm_augment_enrollment_variants(&[(f1_pcm.clone(), "F1.json_enroll0".to_string())]);
    // mahbot-1023: the acceptance arbiter is the CONSTANT
    // VERIFIER_ACCEPTANCE_FLOOR (0.86) — the runtime-calibrated
    // `verifier.threshold` becomes a report-only reference (drift
    // observability) and no longer gates confirmation.
    let verifier_threshold = verifier.threshold;
    let acceptance_floor = crate::audio::voice_verifier::VERIFIER_ACCEPTANCE_FLOOR;
    let b68_grid_idx = SWEEP_BS
        .iter()
        .position(|&b| b == DECISIVE_B)
        .expect("DECISIVE_B in SWEEP_BS");
    let pos24_cell_idx = SWEEP_POSITIONS
        .iter()
        .position(|&p| p == DECISIVE_POS)
        .expect("DECISIVE_POS in SWEEP_POSITIONS");
    let mut variants_raw = Vec::new();
    let mut confirms: Vec<Option<bool>> = Vec::new();
    let mut unmeasurable_variants = Vec::new();
    for (variant_pcm, variant_label) in &f1_variants {
        // Each variant recomputes its streaming-layout mel via the SHARED
        // detector (part of the disclosed shared-detector drift).
        let (variant_mel, variant_mel_len) = bsweep_streaming_mel(variant_pcm, true);
        let (variant_grid, _variant_ring4) = bsweep_score_grid(
            &variant_mel,
            classifier,
            verifier,
            dedup_epsilon,
            "streaming-layout-first-B",
        );
        let decisive_cell = &variant_grid[b68_grid_idx][pos24_cell_idx];
        let confirm = if decisive_cell.measurable {
            let cls_ok = decisive_cell
                .classifier_score
                .is_some_and(|s| s >= super::NO_MATCH_RESET_THRESHOLD);
            // Constant 0.86 floor (mahbot-1023), NOT the runtime-calibrated
            // threshold: product behavior must match the acceptance protocol.
            let ver_ok = decisive_cell
                .verifier_ring4
                .is_some_and(|v| v >= acceptance_floor);
            Some(cls_ok && ver_ok)
        } else {
            None
        };
        let b68_max_rolling_sum = variant_grid[b68_grid_idx]
            .iter()
            .map(|c| c.max_rolling_sum)
            .fold(0.0_f32, f32::max);
        if confirm.is_none() {
            unmeasurable_variants.push(variant_label.clone());
        }
        confirms.push(confirm);
        variants_raw.push(BsweepVariantRaw {
            variant: variant_label.clone(),
            mel_len: variant_mel_len,
            confirm,
            b68_pos24_classifier: decisive_cell.classifier_score,
            b68_ring4_verifier_peak: decisive_cell.verifier_ring4,
            b68_max_rolling_sum,
            grid: variant_grid,
        });
    }
    let (confirm_count, measurable_count, passed_4of5) = bsweep_verdict_passes(&confirms);

    // ── 6. Training-path variants (item 3b) ───────────────────────────────
    // Post-AGC/VAD training-path variants: AGC the F1 raw PCM, gate it with a
    // FRESH detector, then augment the VAD-gated speech (original / speed-down
    // / speed-up conditional on VAD-gated pre-pad ≥ 500 ms / vol-down / pink
    // noise SNR 25 dB seed 0) — the exact vad_segment_and_enroll semantics.
    // The produced-variant count per semantics differs from the raw-level
    // variants in item 3a (raw TTS pre-pad gate vs VAD-gated speech gate).
    let variants_training = bsweep_training_path_scores(f1_pcm, classifier, verifier);
    eprintln!(
        "  B-sweep: training-path variants produced: {}/5 (speed_up gate on VAD-gated speech)",
        variants_training.iter().filter(|v| v.produced).count()
    );

    // ── 7. Sanity anchor (item 5) ─────────────────────────────────────────
    // Read the same-run F1 streaming trajectory from the Phase 8b capture
    // (this sweep runs AFTER Phase 8b, so the same-run VAD state applies).
    let sanity_measured = bsweep_sanity_anchor_score(same_audio);
    eprintln!(
        "  B-sweep: sanity anchor — F1 streaming classifier {} (expected ~{BSWEEP_SANITY_EXPECTED})",
        sanity_measured.map_or_else(|| "unavailable".to_string(), |m| format!("{m:.4}")),
    );

    // ── 8. Weights fingerprints (item 6) ──────────────────────────────────
    let classifier_fingerprint = weights_fingerprint_classifier(classifier.weights_ref());
    let verifier_fingerprint = weights_fingerprint_verifier(verifier);
    eprintln!(
        "  B-sweep: weights fingerprints — classifier {:.16}… verifier {:.16}… (verifier threshold {verifier_threshold:.4})",
        &classifier_fingerprint[..16],
        &verifier_fingerprint[..16],
    );

    // ── 9. Cross-check + trailing post-trim (items 12/13) ─────────────────
    let cross_check = bsweep_cross_check(same_audio, shared_len);
    let trailing_post_trim = bsweep_trailing_post_trim(same_audio);

    let phase_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "  B-sweep: verdict path = shared detector; decisive cell measurable: shared={decisive_cell_measurable_shared}, fresh={decisive_cell_measurable_fresh}; confirms {confirm_count}/{measurable_count} → passed_4of5={passed_4of5}"
    );
    Some(BsweepReport {
        phase_ms,
        f1_mel_len_shared: shared_len,
        f1_mel_len_fresh: fresh_len,
        decisive_cell_measurable_shared,
        decisive_cell_measurable_fresh,
        dedup_epsilon,
        dedup_epsilon_rule,
        dedup_dist_min: dist_min,
        dedup_dist_median: dist_median,
        dedup_dist_max: dist_max,
        grid,
        fresh_grid,
        decisive_ring4,
        variants_raw,
        variants_training,
        confirm_count,
        measurable_count,
        passed_4of5,
        unmeasurable_variants,
        sanity_measured,
        classifier_fingerprint,
        verifier_fingerprint,
        verifier_runtime_threshold: verifier_threshold,
        cross_check,
        trailing_post_trim,
    })
}

impl BsweepReport {
    /// Additive `bsweep` JSON section (mahbot-1022), merged with the
    /// same-run negatives extract (`negatives`).
    #[expect(
        clippy::too_many_lines,
        clippy::needless_pass_by_value,
        reason = "negatives is a fresh extract moved into the report JSON at assembly"
    )]
    fn to_json(&self, negatives: serde_json::Value) -> serde_json::Value {
        // Sweep grid rows: per-B with the cell array in position order.
        let grid_rows = |rows: &[Vec<BsweepCell>], geometry: &str| {
            SWEEP_BS
                .iter()
                .enumerate()
                .map(|(i, &b)| {
                    let cells = rows.get(i).cloned().unwrap_or_default();
                    serde_json::json!({
                        "B": b,
                        "geometry": geometry,
                        "cells": cells.iter().map(bsweep_cell_to_json).collect::<Vec<_>>(),
                    })
                })
                .collect::<Vec<_>>()
        };
        let sweep_grid = grid_rows(&self.grid, "streaming-layout-first-B");
        // Fresh-path dual-report grid (B > fresh mel length → UNMEASURABLE cells).
        let fresh_grid = grid_rows(&self.fresh_grid, "streaming-layout-first-B-fresh-vad");

        // Ring-4 detail at (B=68, position 24).  Outer None = decisive cell
        // not measurable; inner None = ring < 4 → NOT evaluated (JSON null,
        // never a zero-confidence 0.0).
        let (ring4_value, per_window_scores, per_window_rf) = match &self.decisive_ring4 {
            Some((Some(max), wins, fams)) => (
                serde_json::json!(max),
                serde_json::json!(wins),
                serde_json::json!(fams),
            ),
            Some((None, _, _)) | None => (
                serde_json::Value::Null,
                serde_json::json!([]),
                serde_json::json!([]),
            ),
        };

        // Per-variant raw sensitivity: B → { positions: [cells] }.
        let variants_raw: Vec<serde_json::Value> = self
            .variants_raw
            .iter()
            .map(|v| {
                let mut sensitivity = serde_json::Map::new();
                for (i, &b) in SWEEP_BS.iter().enumerate() {
                    let cells = v.grid.get(i).cloned().unwrap_or_default();
                    sensitivity.insert(
                        b.to_string(),
                        serde_json::json!({
                            "positions": cells.iter().map(bsweep_cell_to_json).collect::<Vec<_>>(),
                        }),
                    );
                }
                serde_json::json!({
                    "variant": v.variant,
                    "mel_len": v.mel_len,
                    "confirm": v.confirm,
                    "B68_pos24_classifier": v.b68_pos24_classifier,
                    "B68_ring4_verifier_peak": v.b68_ring4_verifier_peak,
                    "max_rolling_sum_B68": v.b68_max_rolling_sum,
                    "sensitivity": serde_json::Value::Object(sensitivity),
                })
            })
            .collect();

        let variants_training: Vec<serde_json::Value> = self
            .variants_training
            .iter()
            .map(|v| {
                serde_json::json!({
                    "variant": v.variant,
                    "produced": v.produced,
                    "n_windows": v.n_windows,
                    "per_window_classifier": v.per_window_classifier,
                    "per_window_verifier": v.per_window_verifier,
                    "max_rolling_sum": v.max_rolling_sum,
                })
            })
            .collect();

        let sanity_diverged = self
            .sanity_measured
            .is_some_and(|m| (m - BSWEEP_SANITY_EXPECTED).abs() > BSWEEP_SANITY_TOLERANCE);

        serde_json::json!({
            "note": format!(
                "Static-gate (post-fix) condition measurement — NOT today's live \
                 cold-pass behavior (bootstrap-contamination threshold inflation).  \
                 Prior baseline is a 0/20 detection-failure run; this sweep does NOT \
                 reproduce or validate it.  SINGLE-RUN CAVEAT: the verifier is a \
                 {}-member multi-seed ensemble (mahbot-1025), so absolute verifier \
                 values in this section are run-specific (entropy base seed) and \
                 interpretable only via the weights fingerprints below (item 6) — do \
                 not compare verifier magnitudes across runs without first confirming \
                 the fingerprints.",
                crate::audio::voice_verifier::VERIFIER_ENSEMBLE_SEEDS,
            ),
            "phase_ms": self.phase_ms,
            "f1_clip": "F1.json_enroll0",
            "vad_path": {
                "primary": "shared_detector",
                "dual_report": "fresh_detector",
                "f1_mel_len_shared": self.f1_mel_len_shared,
                "f1_mel_len_fresh": self.f1_mel_len_fresh,
                "decisive_cell_measurable_shared": self.decisive_cell_measurable_shared,
                "decisive_cell_measurable_fresh": self.decisive_cell_measurable_fresh,
                "drift_disclosure": "Verdict path uses the SHARED VAD detector (faithful \
                                     to the live pipeline).  The sweep itself advances the \
                                     shared VAD detector's internal state; Phase 9+ negative \
                                     phases run after the sweep and therefore see VAD-state \
                                     drift — accepted and documented per manager pin option b.",
            },
            "dedup": {
                "metric": "relative_l2",
                "epsilon": self.dedup_epsilon,
                "epsilon_rule": self.dedup_epsilon_rule,
                "boundary_frame_distance_distribution": {
                    "min": self.dedup_dist_min,
                    "median": self.dedup_dist_median,
                    "max": self.dedup_dist_max,
                },
                "verdict_driving_definition": "tolerance",
                "note": "Boundary-frame relative-L2 distances across the primary (shared) \
                         streaming-layout mel sequence.  epsilon is derived from the largest \
                         multiplicative gap (bimodal split) or 0.0 (tolerance ≡ bit-exact).  \
                         The labeled primary dedup definition for the reported true-unique \
                         counts is the tolerance definition (frame_relative_l2 <= epsilon).  \
                         NOTE: confirm(variant) itself is purely score-based (classifier at \
                         B=68/position 24 >= 0.316 AND ring-4 verifier peak >= threshold) and \
                         never consumes the true-unique counts; the dedup definitions determine \
                         the window-scoped true-unique frame counts (the real-frame band the \
                         dependent implementation ticket uses), and the per-cell \
                         'dedup_definitions_disagree' flag reports divergence between the two \
                         definitions.",
            },
            "sweep_grid": sweep_grid,
            "sweep_grid_fresh_vad_dual": fresh_grid,
            "ring4": {
                "B68_position24": {
                    "ring4_value": ring4_value,
                    "per_window_scores": per_window_scores,
                    "per_window_rf": per_window_rf,
                    "not_evaluated_convention": "verifier at ring < 4 is JSON null (not \
                                                  evaluated) — NOT a zero-confidence reading",
                },
            },
            "variants_augmentation": "Raw-level F1 variants via \
                                      pcm_augment_enrollment_variants: original / \
                                      speed-down (0.95x) / speed-up (1.05x, conditional on \
                                      >= 500 ms) / vol-down (-3 dB) / pink noise (SNR 25 dB, \
                                      seed = clip index 0).  Same augment set the detection \
                                      test uses (mahbot-932 Fix 5 semantics).",
            "variants_produced_counts": {
                "raw_level_total": self.variants_raw.len(),
                "raw_level_speed_up": self
                    .variants_raw
                    .iter()
                    .filter(|v| v.variant.ends_with("_speed_up"))
                    .count(),
                "training_path_total": self
                    .variants_training
                    .iter()
                    .filter(|v| v.produced)
                    .count(),
                "training_path_speed_up": self
                    .variants_training
                    .iter()
                    .filter(|v| v.produced && v.variant == "speed_up")
                    .count(),
                "note": "Raw-level speed_up gate = raw TTS pre-pad >= 500 ms; \
                         training-path speed_up gate = VAD-gated speech pre-pad >= 500 ms.  \
                         The two counts can differ; the 4/5 verdict rule assumes 5 produced \
                         variants, so the produced counts are stated explicitly.",
            },
            "training_path_note": "Training-path variants (item 3b) run with a FRESH \
                                   detector: AGC F1 raw PCM -> vad_gate_streaming_mel \
                                   (fresh detector) -> augment the VAD-gated speech \
                                   (original / speed-down / speed-up conditional on VAD-gated \
                                   pre-pad >= 500 ms / vol-down / pink noise SNR 25 dB \
                                   seed 0) -> whole-utterance mel -> stride-8 embeddings -> \
                                   parameter-passed full-sequence scoring (mirrors \
                                   vad_segment_and_enroll semantics).",
            "variants_raw": variants_raw,
            "variants_training_path": variants_training,
            "verdict": {
                "classifier_threshold": f64::from(super::NO_MATCH_RESET_THRESHOLD),
                "verifier_acceptance_floor": f64::from(
                    crate::audio::voice_verifier::VERIFIER_ACCEPTANCE_FLOOR,
                ),
                "verifier_runtime_threshold_report_only": self.verifier_runtime_threshold,
                "confirm_count": self.confirm_count,
                "measurable_count": self.measurable_count,
                "passed_4of5": self.passed_4of5,
                "unmeasurable_variants": self.unmeasurable_variants,
                "note": "confirm(variant) = classifier(B=68, position 24) >= 0.316 \
                         (NO_MATCH_RESET_THRESHOLD compile-time constant) AND ring-4 \
                         verifier peak >= VERIFIER_ACCEPTANCE_FLOOR (constant 0.86, \
                         mahbot-1023 — NOT the entropy-seeded runtime-calibrated \
                         threshold, which is reported above as a report-only \
                         reference).  Unmeasurable when the variant's streaming-layout \
                         mel < 68 frames (decisive cell UNMEASURABLE; sequence length \
                         reported).  passes = 4 of 5 confirms over the measurable \
                         variants.",
            },
            "negatives": negatives,
            "sanity_anchor": {
                "expected_approx": BSWEEP_SANITY_EXPECTED,
                "measured_classifier": self.sanity_measured,
                "geometry": "streaming-layout-misaligned-start-3",
                "divergence": sanity_diverged,
                "note": "Same-run F1 streaming classifier score from the Phase 8b \
                         training_clip_comparisons (window start 3 — the misaligned \
                         76-real-frame window, DISTINCT from the trailing post-trim \
                         family; fallback: real_frames 76, else largest-real-frame \
                         window).  Divergence flag = |measured - expected| > 0.05.",
            },
            "weights_fingerprints": {
                "classifier": {
                    "sha256": self.classifier_fingerprint,
                    "bytes_hashed": "sha2-256 over f32 LE bytes of weight vectors in field \
                                     order conv1_weight..fc_bias + bn_eps + arch; BN running \
                                     statistics INCLUDED; NO_MATCH_RESET_THRESHOLD is a \
                                     compile-time constant (0.316), not hashed",
                },
                "verifier": {
                    "sha256": self.verifier_fingerprint,
                    "bytes_hashed": "sha2-256 over f32 LE bytes of conv_weight, conv_bias, \
                                     fc_weight, fc_bias, then runtime-calibrated threshold",
                    "runtime_threshold": self.verifier_runtime_threshold,
                },
            },
            "cross_check": self.cross_check,
            "trailing_post_trim": self.trailing_post_trim,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// mahbot-1023: enrolled-speaker benchmark phase (Phase 8d)
// ═══════════════════════════════════════════════════════════════════════════

/// Per-variant enrolled-speaker result (mahbot-1023 item 6).
#[derive(Clone)]
#[allow(clippy::struct_excessive_bools)]
struct EnrolledVariantResult {
    variant: String,
    /// End-to-end detection through the real streaming cold pass.
    detected: bool,
    /// Scoring path that produced the detection, reclassified into the
    /// enrolled-phase mechanism taxonomy (mahbot-1024): "burst" /
    /// "burst_continuation" / "segment_end_pass" / "unexpected" (see
    /// [`reclassify_detection_path`]).  "burst_continuation" (burst-created
    /// candidate confirmed by main-loop continuation) is the PRIMARY
    /// expected mechanism; "unexpected" (main-loop-created candidate) is
    /// the genuine regression signal.  None when not detected.
    detection_path: Option<String>,
    /// Raw detection source ("burst" / "segment_end_pass" / "other") — the
    /// un-reclassified window type where the confirmation fired, kept for
    /// report transparency alongside the reclassified `detection_path`.
    detection_source: Option<String>,
    /// confirm(variant) per the acceptance protocol: classifier gate
    /// (max rolling sum ≥ MIN_CLASSIFIER_THRESHOLD) AND the RING-4 verifier
    /// peak at the burst's position 24 (the 4th scored frame's
    /// max_verifier_score — the live analogue of the B-sweep decisive cell).
    confirm: bool,
    /// Ring-4 verifier peak at the burst's position 24 (the 4th scored
    /// frame).  0.0 when fewer than 4 frames were scored.
    ring4_verifier_at_pos24: f32,
    /// Detection latency ms (CPU wall-clock of the feed loop; None when not
    /// detected).  NOTE: the feed loop processes faster than real-time, so
    /// this is a processing-cost proxy — the AUDIO position of the detection
    /// is given by `burst_sweep_buffer_len` × 10 ms (mel model stride).
    latency_ms: Option<f64>,
    /// Peak verifier score achieved by any candidate this variant (the
    /// session-lifetime peak, which can exceed the ring-4 sample when later
    /// clean windows confirm the burst-created candidate).
    verifier_peak: f32,
    /// Per-frame classifier window scores `[total_score, rolling_sum,
    /// effective_threshold]`.
    per_frame_scores: Vec<[f32; 3]>,
    /// Per-frame verifier scores (parallel to `per_frame_scores`).
    verifier_score_trajectory: Vec<f32>,
    /// Per-frame mel-window start positions (parallel to `per_frame_scores`).
    per_frame_window_start: Vec<usize>,
    /// Per-frame mel-buffer lengths (parallel to `per_frame_scores`).
    per_frame_mel_buffer_len: Vec<usize>,
    /// Per-frame window geometry classes (parallel to `per_frame_scores`).
    per_frame_geometry: Vec<super::WindowGeometry>,
    /// Per-frame adaptive-threshold modes (parallel to `per_frame_scores`).
    per_frame_adaptive_mode: Vec<super::AdaptiveFrameMode>,
    /// Per-frame candidate lifecycle states (parallel to `per_frame_scores`).
    per_frame_candidate_state: Vec<super::CandidateFrameState>,
    /// Live burst-trigger buffer length (None when the burst never ran).
    burst_sweep_buffer_len: Option<usize>,
    /// Synchronous burst sweep wall-clock (ms; None when the burst never ran).
    burst_wall_clock_ms: Option<f64>,
    /// Whether the segment-end pass ran this variant.
    segment_end_pass_fired: bool,
    /// Whether the adaptive bootstrap persisted across the whole utterance.
    adaptive_bootstrap_persisted: bool,
    /// Miss verdict for non-detected variants (see [`MissVerdict`]).
    miss_verdict: Option<MissVerdict>,
}

/// Run the enrolled-speaker benchmark phase (mahbot-1023 item 6).
///
/// Positive set: F1's 5 augmentation variants (`_original`, `_speed_down`
/// (0.95×), `_speed_up` (1.05×, conditional on ≥500 ms pre-pad), `_vol_down`
/// (−3 dB), `_noise` (pink noise ~25 dB SNR)) — labeled prominently as
/// IN-SAMPLE / TRAINING-SIDE CONTROL data (user-approved "one speaker first"
/// direction, NOT a generalization measure; cross-speaker generalization is
/// a later classifier phase).
///
/// Measures end-to-end detection (classifier gate + verifier ≥ 0.86) through
/// the REAL streaming cold pass — fresh [`PipelineCtx`] per variant, no
/// warm-up, fresh adaptive state that bootstraps from the first real frames,
/// cold-start verifier warm-up included — exactly the production post-silence
/// start.  The deferral (deferred burst + segment-end pass) is exercised for
/// real through `handle_wake_word_detection`; the synchronous burst stall is
/// measured per variant (the drop-counter proxy, manager pin 4).
///
/// Returns `None` when the F1 training clip is absent.
#[expect(clippy::too_many_lines)]
fn run_enrolled_speaker_phase(
    train_clips: &[(Vec<f32>, String)],
    classifier: &WakeWordClassifier,
    verifier: &VoiceVerifier,
) -> Option<serde_json::Value> {
    let start = Instant::now();
    let (f1_pcm, _f1_label) = train_clips.iter().find(|(_, l)| l == "F1.json_enroll0")?;
    let f1_variants =
        pcm_augment_enrollment_variants(&[(f1_pcm.clone(), "F1.json_enroll0".to_string())]);

    // ── Set classifier + verifier in global state (the streaming pipeline
    //    reads them from voice_state) ───────────────────────────────────────
    super::set_classifier_weights(classifier.weights_ref().clone());
    super::set_verifier(verifier.clone());

    let mut variants: Vec<EnrolledVariantResult> = Vec::new();
    for (variant_pcm, variant_label) in &f1_variants {
        // Cold pass: fresh PipelineCtx per variant, no consume_warmup, fresh
        // AdaptiveThresholdState (matches production's post-silence start).
        let mut ctx = super::PipelineCtx::new();
        let result = run_streaming_detection(variant_pcm, &mut ctx);
        let pv = build_per_variant_result(variant_label, &result, &ctx, false, 0, verifier);
        // confirm(variant) per the acceptance protocol: the ring-4 verifier
        // peak at the burst's position 24 (the 4th scored frame's
        // max_verifier_score) — the live analogue of the B-sweep decisive
        // cell.  NOTE: at the live trigger geometry (B = 72–80) position 0 is
        // a full 76-frame window (not padded), whose embedding can pull the
        // ring-4 sample below 0.86 even when later clean windows confirm —
        // that divergence is exactly what the ≥3-run protocol measures; the
        // session `verifier_peak` is reported alongside for comparison.
        let ring4_verifier_at_pos24 = pv.verifier_score_trajectory.get(3).copied().unwrap_or(0.0);
        let confirm = pv.max_rolling_sum >= MIN_CLASSIFIER_THRESHOLD
            && ring4_verifier_at_pos24 >= crate::audio::voice_verifier::VERIFIER_ACCEPTANCE_FLOOR;
        variants.push(EnrolledVariantResult {
            variant: variant_label.clone(),
            detected: result.detected,
            detection_path: reclassify_detection_path(
                pv.detection_path.as_deref(),
                pv.candidate_created_path.as_deref(),
            ),
            detection_source: pv.detection_path.clone(),
            confirm,
            ring4_verifier_at_pos24,
            latency_ms: result.latency_ms,
            verifier_peak: pv.verifier_score,
            per_frame_scores: pv.per_frame_scores.clone(),
            verifier_score_trajectory: pv.verifier_score_trajectory.clone(),
            per_frame_window_start: pv.per_frame_window_start.clone(),
            per_frame_mel_buffer_len: pv.per_frame_mel_buffer_len.clone(),
            per_frame_geometry: pv.per_frame_geometry.clone(),
            per_frame_adaptive_mode: pv.per_frame_adaptive_mode.clone(),
            per_frame_candidate_state: pv.per_frame_candidate_state.clone(),
            burst_sweep_buffer_len: pv.burst_sweep_buffer_len,
            burst_wall_clock_ms: ctx.instrumentation.burst_wall_clock_ms,
            segment_end_pass_fired: pv.segment_end_pass_fired,
            adaptive_bootstrap_persisted: pv.adaptive_bootstrap_persisted,
            miss_verdict: if result.detected {
                None
            } else {
                Some(classify_miss(&pv))
            },
        });
        info!(
            "  Enrolled variant {}: {} — {} (mechanism: {}, source: {}, latency: {:?}ms, verifier peak: {:.4}, ring4@pos24: {:.4}, confirm: {})",
            variants.len(),
            variant_label,
            if result.detected { "DETECTED" } else { "miss" },
            variants
                .last()
                .and_then(|v| v.detection_path.as_deref())
                .unwrap_or("n/a"),
            pv.detection_path.as_deref().unwrap_or("n/a"),
            result.latency_ms,
            pv.verifier_score,
            ring4_verifier_at_pos24,
            confirm,
        );
    }

    // ── Per-member speed_down peaks (mahbot-1025 variance-reduction gate) ──
    // For every ensemble member (including the primary), measure the member's
    // STANDALONE verifier peak on the F1 speed_down variant through the same
    // streaming cold pass used for the acceptance metric.  The fraction of
    // TRAINED members whose standalone peak falls below
    // VERIFIER_ACCEPTANCE_FLOOR is the per-run single-seed speed_down
    // miss-probability estimate (gate ≤0.15, target band ~0.04–0.10).  The
    // ensemble mean (speed_down_peak above) is the acceptance metric; the
    // member distribution is the variance-reduction evidence.
    //
    // Denominator semantics: only TRAINED members are counted.  Members share
    // training data (same minimum-window guard), so they train/fail together
    // — but if one were untrained, its standalone peak would be a guaranteed
    // 1.0 (accept-all) and counting it would silently inflate the denominator
    // with a guaranteed pass.  Untrained members are excluded and logged; a
    // variant with zero trained members fails closed (miss fraction 1.0).
    let mut speed_down_member_peaks: Vec<f32> = Vec::new();
    let mut speed_down_member_untrained: usize = 0;
    let speed_down_variant = f1_variants.iter().find(|(_, l)| l.ends_with("_speed_down"));
    if let Some((spd_pcm, _)) = speed_down_variant {
        for member_idx in 0..verifier.ensemble_size() {
            let member_verifier = verifier.member_only(member_idx);
            if !member_verifier.is_trained() {
                speed_down_member_untrained += 1;
                warn!(
                    "speed_down member gate: member {member_idx} is UNTRAINED — \
                     excluded from the miss-fraction denominator (members share \
                     training data, so this indicates an anomaly; mahbot-1025)",
                );
                continue;
            }
            super::set_verifier(member_verifier.clone());
            let mut mctx = super::PipelineCtx::new();
            let mres = run_streaming_detection(spd_pcm, &mut mctx);
            let mpv = build_per_variant_result(
                "speed_down_member",
                &mres,
                &mctx,
                false,
                0,
                &member_verifier,
            );
            speed_down_member_peaks.push(mpv.verifier_score);
        }
        // Restore the full ensemble verifier for the remaining phases.
        super::set_verifier(verifier.clone());
    }
    let speed_down_member_miss_fraction: f32 = if speed_down_member_peaks.is_empty() {
        // Not measurable: no speed_down variant (report 0.0 as before) OR the
        // variant existed but every member was untrained (fail-closed 1.0).
        if speed_down_member_untrained > 0 {
            warn!(
                "speed_down member gate: no trained members to measure — miss \
                 fraction reported as 1.0 (fail-closed; mahbot-1025)",
            );
            1.0
        } else {
            0.0
        }
    } else {
        let below = speed_down_member_peaks
            .iter()
            .filter(|&&p| p < crate::audio::voice_verifier::VERIFIER_ACCEPTANCE_FLOOR)
            .count();
        below as f32 / speed_down_member_peaks.len() as f32
    };
    if !speed_down_member_peaks.is_empty() {
        info!(
            "Verifier ensemble speed_down member peaks: {:?} — miss fraction {:.3} \
             over {n_trained} trained member(s) (gate ≤0.15, mahbot-1025)",
            speed_down_member_peaks,
            speed_down_member_miss_fraction,
            n_trained = speed_down_member_peaks.len(),
        );
    }

    // ── Aggregate acceptance metrics ────────────────────────────────────────
    let total = variants.len();
    let detected_live = variants.iter().filter(|v| v.detected).count();
    let confirmed = variants.iter().filter(|v| v.confirm).count();
    // F1 no-regression control (acceptance criterion 2): the original
    // enrolled variant must be detected in every fresh run.
    let f1_original_detected = variants
        .iter()
        .find(|v| v.variant.ends_with("_original"))
        .is_some_and(|v| v.detected);
    // mahbot-1025: F1_noise explicit measurement — the F1 noise-augmented
    // variant's verifier peak must stay ≥ the 0.86 floor in every run (it is
    // borderline 0.858–0.932 in the archive and the negative-corpus expansion
    // must not push it below floor; measured explicitly, never assumed).
    let f1_noise_verifier_peak = variants
        .iter()
        .find(|v| v.variant.ends_with("_noise"))
        .map(|v| v.verifier_peak);
    let burst_path = variants
        .iter()
        .filter(|v| v.detection_path.as_deref() == Some("burst"))
        .count();
    let burst_cont_path = variants
        .iter()
        .filter(|v| v.detection_path.as_deref() == Some("burst_continuation"))
        .count();
    let pass_path = variants
        .iter()
        .filter(|v| v.detection_path.as_deref() == Some("segment_end_pass"))
        .count();
    let unexpected_path = variants
        .iter()
        .filter(|v| v.detection_path.as_deref() == Some("unexpected"))
        .count();
    // PRIMARY expected mechanism: burst-family detections (burst sweep
    // directly, or burst-created candidate confirmed by main-loop
    // continuation) before the segment-end pass.
    let primary_mechanism = burst_path + burst_cont_path;
    let burst_latencies: Vec<f64> = variants
        .iter()
        .filter(|v| v.detection_path.as_deref() == Some("burst"))
        .filter_map(|v| v.latency_ms)
        .collect();
    let burst_family_latencies: Vec<f64> = variants
        .iter()
        .filter(|v| {
            matches!(
                v.detection_path.as_deref(),
                Some("burst" | "burst_continuation")
            )
        })
        .filter_map(|v| v.latency_ms)
        .collect();
    let burst_stalls: Vec<f64> = variants
        .iter()
        .filter_map(|v| v.burst_wall_clock_ms)
        .collect();
    let live_trigger_bs: Vec<usize> = variants
        .iter()
        .filter_map(|v| v.burst_sweep_buffer_len)
        .collect();
    let mean = |xs: &[f64]| -> Option<f64> {
        if xs.is_empty() {
            None
        } else {
            Some(xs.iter().copied().sum::<f64>() / xs.len() as f64)
        }
    };

    // ── Near-miss canaries (mahbot-1024 item 5) ────────────────────────────
    // Investigation triggers, NOT hard gates: firing means the run deserves
    // review, never an automatic pass/fail.  Per-run reported.
    let mut canaries_fired: Vec<&'static str> = Vec::new();
    let mut confirmation_margin_low: Vec<serde_json::Value> = Vec::new();
    let mut speed_down_peak: Option<f32> = None;
    for v in &variants {
        if v.variant.ends_with("_speed_down") {
            speed_down_peak = Some(v.verifier_peak);
        }
        // Canary: any per-run confirmation verifier margin below ~0.90
        // (confirmed detection whose verifier peak sat in [0.86, 0.90)).
        if v.detected
            && v.verifier_peak >= crate::audio::voice_verifier::VERIFIER_ACCEPTANCE_FLOOR
            && v.verifier_peak < 0.90
        {
            confirmation_margin_low.push(serde_json::json!({
                "variant": v.variant,
                "verifier_peak": v.verifier_peak,
                "margin_above_floor": v.verifier_peak
                    - crate::audio::voice_verifier::VERIFIER_ACCEPTANCE_FLOOR,
            }));
        }
    }
    if !confirmation_margin_low.is_empty() {
        canaries_fired.push("confirmation_verifier_margin_below_0_90");
    }
    // Canary: speed_down variant peak below ~0.90.
    let speed_down_below_0_90 = speed_down_peak.is_some_and(|p| p < 0.90);
    if speed_down_below_0_90 {
        canaries_fired.push("speed_down_peak_below_0_90");
    }
    // mahbot-1025: F1_noise explicit canary — verifier peak below the 0.86
    // acceptance floor (borderline in the archive; must be re-measured every
    // run, never assumed).
    if f1_noise_verifier_peak
        .is_some_and(|p| p < crate::audio::voice_verifier::VERIFIER_ACCEPTANCE_FLOOR)
    {
        canaries_fired.push("f1_noise_verifier_below_floor");
    }

    let per_variant_json: Vec<serde_json::Value> = variants
        .iter()
        .map(|v| {
            serde_json::json!({
                "variant": v.variant,
                "detected": v.detected,
                "detection_path": v.detection_path,
                "detection_source": v.detection_source,
                "confirm": v.confirm,
                "ring4_verifier_at_pos24": v.ring4_verifier_at_pos24,
                "latency_ms": v.latency_ms,
                "verifier_peak": v.verifier_peak,
                "classifier_window_scores": v.per_frame_scores,
                "verifier_score_trajectory": v.verifier_score_trajectory,
                "per_frame_window_start": v.per_frame_window_start,
                "per_frame_mel_buffer_len": v.per_frame_mel_buffer_len,
                "per_frame_geometry": v
                    .per_frame_geometry
                    .iter()
                    .map(|g| g.as_str())
                    .collect::<Vec<_>>(),
                "per_frame_adaptive_mode": v
                    .per_frame_adaptive_mode
                    .iter()
                    .map(|m| m.as_str())
                    .collect::<Vec<_>>(),
                "per_frame_candidate_state": v
                    .per_frame_candidate_state
                    .iter()
                    .map(|c| c.as_str())
                    .collect::<Vec<_>>(),
                "burst_sweep_buffer_len": v.burst_sweep_buffer_len,
                "burst_wall_clock_ms": v.burst_wall_clock_ms,
                "segment_end_pass_fired": v.segment_end_pass_fired,
                "adaptive_bootstrap_persisted": v.adaptive_bootstrap_persisted,
                "miss_verdict": v.miss_verdict.map(MissVerdict::as_str),
            })
        })
        .collect();

    let phase_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "  Enrolled-speaker phase: {detected_live}/{total} detected end-to-end, \
         {confirmed}/{total} confirmed (strict ring-4 formula, report-only), \
         paths: burst={burst_path} burst_continuation={burst_cont_path} \
         segment_end_pass={pass_path} unexpected={unexpected_path}",
    );
    if unexpected_path > 0 {
        warn!(
            "Enrolled-speaker phase: {unexpected_path} detection(s) via the 'unexpected' \
             path (main-loop-created candidate — no burst-created candidate to continue) — \
             genuine unexpected-path signal, must be investigated",
        );
    }
    if !canaries_fired.is_empty() {
        warn!(
            "Enrolled-speaker phase near-miss canaries fired: {} — investigation \
             triggers, not a hard gate",
            canaries_fired.join(", "),
        );
    }

    Some(serde_json::json!({
        "note": "IN-SAMPLE / TRAINING-SIDE CONTROL data (user-approved 'one speaker \
                 first' direction) — NOT a generalization measure.  Measures end-to-end \
                 detection (classifier gate + verifier >= VERIFIER_ACCEPTANCE_FLOOR 0.86) \
                 through the REAL streaming cold pass: fresh PipelineCtx per variant, no \
                 warm-up, fresh adaptive bootstrap, cold-start verifier warm-up included.  \
                 Acceptance is re-scoped (mahbot-1024) to the MEASURED PRODUCTION \
                 MECHANISM: a burst-created candidate confirmed mid-utterance at verifier \
                 >= 0.86, whether confirmed inside the burst window ('burst') or by \
                 main-loop continuation of the burst-created candidate \
                 ('burst_continuation' — the PRIMARY expected path, 12/14 of the \
                 archived final-code detections, at 760-790 ms from wake-word onset).  \
                 The strict confirm(variant) formula (ring-4 verifier at the burst's \
                 position 24, below) and the batch B-sweep readings are report-only \
                 diagnostics and do NOT gate acceptance.",
        "f1_clip": "F1.json_enroll0",
        "phase_ms": phase_ms,
        "variants_produced": {
            "total": total,
            "raw_level_speed_up": variants.iter().filter(|v| v.variant.ends_with("_speed_up")).count(),
            "raw_level_total": variants.len(),
            "training_path_speed_up": null,
            "note": "Produced-variant count per semantics: raw-level speed_up gate = raw \
                     TTS pre-pad >= 500 ms (this phase's count); the training-path gate \
                     (VAD-gated speech pre-pad) is reported in the bsweep section.  The \
                     5-variant set assumes speed_up was produced; if it was skipped the \
                     produced count is stated explicitly.",
        },
        "acceptance": {
            "total_variants": total,
            "detected_live": detected_live,
            "detected_live_frac": if total > 0 { detected_live as f64 / total as f64 } else { 0.0 },
            "f1_original_detected": f1_original_detected,
            "confirmed": confirmed,
            "confirmed_frac": if total > 0 { confirmed as f64 / total as f64 } else { 0.0 },
            "target": "mean TP >= 4/5 (0.80) across >= 3 FRESH runs of the final staged \
                       code — per-run end-to-end detection through the measured \
                       production mechanism (detected_live), NOT the strict ring-4 \
                       formula",
            "strict_ring4_formula_report_only": "confirm(variant) = classifier gate \
                       (max_rolling_sum >= 2.13) AND ring-4 verifier peak at the burst's \
                       position 24 >= 0.86 (the live analogue of the B-sweep decisive \
                       cell).  Kept visible as a report-only diagnostic: at the live \
                       trigger geometry (B = 72-80) the burst's position-0 window is a \
                       full 76-frame window (not padded) and its embedding can pull the \
                       ring-4 sample below 0.86 even when the pipeline detects — the \
                       divergence is the measurement artifact the re-scope documents, \
                       NOT a mechanism failure.",
            "effective_verifier_threshold": crate::audio::voice_verifier::VERIFIER_ACCEPTANCE_FLOOR,
            "speed_down_member_peaks": speed_down_member_peaks,
            "speed_down_member_miss_fraction": speed_down_member_miss_fraction,
            // mahbot-1025: explicit F1_noise measurement (verifier peak must
            // stay ≥ 0.86 every run — borderline in the archive).
            "f1_noise_verifier_peak": f1_noise_verifier_peak,
            "f1_noise_above_floor": f1_noise_verifier_peak
                .is_some_and(|p| p >= crate::audio::voice_verifier::VERIFIER_ACCEPTANCE_FLOOR),
            "note": "detected_live = end-to-end through the real cold pass (the \
                     acceptance metric).  confirmed = strict ring-4 formula (report-only \
                     diagnostic).  f1_original_detected = F1 no-regression control \
                     (criterion 2: must be detected in every fresh run).  \
                     speed_down_member_peaks = each ensemble member's STANDALONE \
                     speed_down verifier peak (mahbot-1025); the miss fraction is the \
                     per-run single-seed miss-probability estimate (gate ≤0.15, target \
                     band ~0.04–0.10).  f1_noise_verifier_peak = the F1 noise variant's \
                     verifier peak, measured explicitly every run (must be ≥ 0.86).",
        },
        "paths": {
            "burst": burst_path,
            "burst_continuation": burst_cont_path,
            "segment_end_pass": pass_path,
            "unexpected": unexpected_path,
            "primary_mechanism": primary_mechanism,
            "note": "burst = deferred burst sweep confirmed directly; \
                     burst_continuation = burst-created candidate confirmed by \
                     main-loop continuation BEFORE the segment-end pass (the PRIMARY \
                     expected mechanism, mahbot-1024); segment_end_pass = boundary \
                     fallback; unexpected = main-loop-created candidate (no \
                     burst-created candidate to continue) — the genuine unexpected-path \
                     signal, flag + investigate, never accept silently.  Acceptance \
                     requires mean TP >= 4/5 across >= 3 fresh runs via the primary \
                     mechanism family (burst + burst_continuation), before the \
                     segment-end pass.",
        },
        "latency": {
            "burst_path_ms": burst_latencies,
            "burst_path_mean_ms": mean(&burst_latencies),
            "burst_family_ms": burst_family_latencies,
            "burst_family_mean_ms": mean(&burst_family_latencies),
            "burst_path_target": "<= ~1.0 s from speech onset",
            "audio_position_of_burst_ms": live_trigger_bs
                .iter()
                .map(|b| b * 10)
                .collect::<Vec<_>>(),
            "note": "latency_ms is the CPU wall-clock of the feed loop (the benchmark \
                     feeds audio faster than real-time) — a processing-cost proxy, NOT \
                     the 'from speech onset' audio latency.  The AUDIO position of the \
                     burst is burst_sweep_buffer_len x 10 ms (mel model stride): the \
                     live trigger fires ~680-800 ms into the utterance (~56% through \
                     'mahbot' for F1 at B=68-80), i.e. as the user finishes speaking \
                     it, NOT instantly — the intentional UX framing.  The <= ~1.0 s \
                     bound is judged on the audio position; observed confirmation \
                     760-790 ms from wake-word onset.  burst_family_ms covers the \
                     primary-mechanism detections (burst + burst_continuation).",
        },
        "burst_stall": {
            "wall_clock_ms": burst_stalls,
            "max_ms": burst_stalls.iter().copied().fold(0.0_f64, f64::max),
            "note": "Synchronous burst sweep stall measured through the live pipeline \
                     (AGC/VAD/block_in_place overhead included).  Worst case ~44-135 ms \
                     (up to 9 ONNX calls), ~20-60 ms on the confirming path (4 calls); \
                     the 1.024 s mic channel absorbs it.  No async scoring path is used.  \
                     DROPPED_CHUNKS is only incremented on the real-mic channel — the \
                     benchmark feeds audio directly, so the no-drop criterion is an \
                     operational check on the live mic channel, stated as such (manager \
                     pin 4).",
        },
        "live_trigger_geometry": {
            "burst_sweep_buffer_lens": live_trigger_bs,
            "note": "Actual live burst-trigger buffer lengths per variant (the first \
                     flush-aligned B >= 68).  Reported per run so the acceptance review \
                     can see the live geometry instead of extrapolating from the B=68 \
                     static-gate grid (manager pin 2).",
        },
        "near_miss_canaries": {
            "confirmation_verifier_margin_below_0_90": confirmation_margin_low,
            "speed_down_peak": speed_down_peak,
            "speed_down_below_0_90": speed_down_below_0_90,
            "fired": canaries_fired,
            "note": "Investigation triggers (mahbot-1024 item 5), NOT hard gates: any \
                     fired canary means the run deserves review before acceptance.  \
                     'confirmation_verifier_margin_below_0_90' = a confirmed detection \
                     whose verifier peak sat in [0.86, 0.90); 'speed_down_peak_below_0_90' \
                     = the binding speed_down variant's verifier peak dipped below 0.90 \
                     (TWO observed speed_down sub-floor peaks: 0.7886 run 20260801-061348 \
                     and 0.8090 run 20260801-085648 — not one); \
                      'f1_noise_verifier_below_floor' = the F1 noise variant's verifier \
                      peak fell below the 0.86 acceptance floor (mahbot-1025 explicit \
                      measurement).  Negative-corpus canaries (classifier gate \
                     crossings, negative-verifier margin-to-floor) are reported in the \
                     safety_gate section; the per-run verifier weights fingerprint is in \
                     the bsweep section.",
        },
        "per_variant": per_variant_json,
    }))
}

/// Compute the mahbot-1023 safety gate over the negative/confusable set.
///
/// The deferral adds in-distribution deferred windows (burst + boundary
/// double-scoring) whose false-positive impact must be MEASURED, not assumed.
/// Baseline (mahbot-1022): 9/59 classifier gate crossings, 0 end-to-end false
/// accepts.  The verifier is the false-accept backstop; the confusable
/// `day mahbot_s2` peak 0.7587 sits 0.1013 below the 0.86 floor.  Because the
/// verifier is entropy-seeded, an in-distribution confusable may exceed the
/// floor on some runs — the report distinguishes true false accepts (detected)
/// from boundary-crossing confusables (verifier peak in [0.86, 1.0] but no
/// end-to-end accept).
#[expect(clippy::too_many_lines)]
fn safety_gate(all_neg_pv: &[(&PerVariantResult, String)]) -> serde_json::Value {
    let acceptance_floor = crate::audio::voice_verifier::VERIFIER_ACCEPTANCE_FLOOR;
    let total = all_neg_pv.len();
    let gate_crossings = all_neg_pv
        .iter()
        .filter(|(pv, _)| pv.max_rolling_sum >= MIN_CLASSIFIER_THRESHOLD)
        .count();
    let false_accepts: Vec<(&str, String)> = all_neg_pv
        .iter()
        .filter(|(pv, _)| pv.detected)
        .map(|(pv, cat)| (pv.variant.as_str(), cat.clone()))
        .collect();
    let boundary_crossers: Vec<serde_json::Value> = all_neg_pv
        .iter()
        .filter(|(pv, _)| {
            pv.verifier_score >= crate::audio::voice_verifier::VERIFIER_ACCEPTANCE_FLOOR
        })
        .map(|(pv, cat)| {
            serde_json::json!({
                "variant": pv.variant,
                "category": cat,
                "verifier_peak": pv.verifier_score,
                "detected": pv.detected,
                "gate_crossed": pv.max_rolling_sum >= MIN_CLASSIFIER_THRESHOLD,
            })
        })
        .collect();
    // mahbot-1023 review fix: `pv.verifier_score` is the CANDIDATE-gated peak
    // (`peak_verifier_score`, updated only while a candidate is active) — it
    // collapses to 0.0 when no negative crosses the classifier gate, so it
    // does NOT measure the verifier's negative distribution under the
    // deferred geometry.  `verifier_score_trajectory` carries the raw
    // per-frame verifier score (computed for EVERY scored frame once the ring
    // passes VERIFIER_WARMUP_EMBEDDINGS, independent of candidate state), so
    // aggregating it actually re-measures the 0.86 floor's FP margin as the
    // ticket requires (item 7: must be measured, not assumed).  Acceptance
    // accounting stays per-utterance (manager pin 7): a per-frame floor
    // crossing is a boundary-crossing confusable, NOT a true false accept —
    // the classifier gate must also pass for end-to-end detection.
    let mut per_frame_max: f32 = 0.0;
    let mut per_frame_total: usize = 0;
    let mut per_frame_above_floor: usize = 0;
    let mut boundary_crossing_confusables: Vec<serde_json::Value> = Vec::new();
    // mahbot-1024 near-miss canary: negative per-frame verifier max within
    // 0.20 of the floor (margin-to-floor < 0.20) — investigation trigger,
    // NOT a hard gate.
    let mut negative_margin_low: Vec<serde_json::Value> = Vec::new();
    for (pv, cat) in all_neg_pv {
        let mut utt_max = 0.0_f32;
        let mut utt_above_floor = false;
        for &v in &pv.verifier_score_trajectory {
            per_frame_total += 1;
            per_frame_max = per_frame_max.max(v);
            utt_max = utt_max.max(v);
            if v >= acceptance_floor {
                per_frame_above_floor += 1;
                utt_above_floor = true;
            }
        }
        if utt_above_floor {
            boundary_crossing_confusables.push(serde_json::json!({
                "variant": pv.variant,
                "category": cat,
                "per_frame_verifier_max": utt_max,
                "detected": pv.detected,
                "gate_crossed": pv.max_rolling_sum >= MIN_CLASSIFIER_THRESHOLD,
            }));
        }
        if utt_max > acceptance_floor - 0.20 {
            negative_margin_low.push(serde_json::json!({
                "variant": pv.variant,
                "category": cat,
                "per_frame_verifier_max": utt_max,
                "margin_to_floor": acceptance_floor - utt_max,
            }));
        }
    }
    // mahbot-1024 near-miss canaries (investigation triggers, not hard gates).
    let mut neg_canaries_fired: Vec<&'static str> = Vec::new();
    if gate_crossings > 0 {
        neg_canaries_fired.push("classifier_gate_crossing_on_negative_corpus");
    }
    if !negative_margin_low.is_empty() {
        neg_canaries_fired.push("negative_verifier_margin_to_floor_below_0_20");
    }
    serde_json::json!({
        "note": "False-positive impact of the deferred in-distribution windows \
                 (deferred burst + boundary fallback double-scoring) on the \
                 negative/confusable set.  Baseline (mahbot-1022, pre-deferral): \
                 9/59 classifier gate crossings, 0 end-to-end false accepts; \
                 confusable 'day mahbot_s2' verifier peak 0.7587 (margin 0.1013 \
                 below the 0.86 floor).  The gate crossing count is EXPECTED to \
                 change under the deferral — it must be measured, not assumed.  \
                 Per-utterance accounting (manager pin 7): the acceptance \
                 denominator is utterances, not frames.  The per-frame negative \
                 verifier aggregation (mahbot-1023 review fix) uses the raw \
                 per-frame verifier scores from the trajectories — NOT the \
                 candidate-gated peak, which is vacuous when no negative crosses \
                 the classifier gate — so the 0.86 floor's FP margin is actually \
                 re-measured under the new burst geometry.",
        "total_negatives": total,
        "classifier_gate_crossings": gate_crossings,
        "baseline_gate_crossings": 9,
        "end_to_end_false_accepts": false_accepts.len(),
        "false_accept_list": false_accepts,
        "baseline_false_accepts": 0,
        "candidate_gated_peak": {
            "boundary_crossing_confusables": boundary_crossers,
            "max_negative_verifier_peak": all_neg_pv
                .iter()
                .map(|(pv, _)| pv.verifier_score)
                .fold(0.0_f32, f32::max),
            "note": "Legacy candidate-gated metrics (baseline comparability).  \
                     Vacuous when no negative crosses the classifier gate — see \
                     per_frame_negative_verifier for the real distribution.",
        },
        "per_frame_negative_verifier": {
            "max": per_frame_max,
            "frames_total": per_frame_total,
            "frames_above_floor": per_frame_above_floor,
            "boundary_crossing_confusables": boundary_crossing_confusables,
            "note": "Aggregated from verifier_score_trajectory (raw per-frame \
                     verifier scores, not candidate-gated).",
        },
        "verifier_acceptance_floor": acceptance_floor,
        "margin_to_floor": (per_frame_max - acceptance_floor).abs(),
        "acceptance": "<= 2/59 false accepts (5%) across >= 3 runs; every run with a \
                       false accept must be investigated before the deferral counts \
                       as enabled — no FA-bound breach may be silently tolerated.  \
                       Boundary-crossing confusables (verifier >= floor without a \
                       classifier-gate crossing) are reported separately and do NOT \
                       count as false accepts.",
        "near_miss_canaries": {
            "classifier_gate_crossing_on_negative_corpus": gate_crossings,
            "negative_verifier_margin_to_floor_below_0_20": negative_margin_low,
            "fired": neg_canaries_fired,
            "note": "Investigation triggers (mahbot-1024 item 5), NOT hard gates.  \
                     'classifier_gate_crossing_on_negative_corpus' = any negative \
                     variant crossed the classifier gate (baseline 9/59, mahbot-1022 \
                     pre-deferral — expected to fire most runs; the count is measured \
                     per run, not assumed).  'negative_verifier_margin_to_floor_below_0_20' \
                     = a negative's per-frame verifier max came within 0.20 of the \
                     0.86 floor (archive-worst: 0.8396 run 20260801-085346, margin 0.0204; \
                     fresh-run worst: 0.7295 run 20260801-090824, margin 0.1305; fresh-run \
                     margin distribution 0.7901 / 0.2680 / 0.1305 across runs 090605 / \
                     090719 / 090824 — the 0.6833 reading is archive second-worst, NOT \
                     the floor-relevant peak).  Positive-side canaries are in the \
                     enrolled_speaker section; the per-run verifier weights \
                     fingerprint is in the bsweep section.",
        },
    })
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

    // ── Verifier-only positive expansion (mahbot-1025) ─────────────────
    // The verifier's tiny positive pool (~30-55 windows from the 6 train
    // clips) is the root of its seed-driven calibration variance.  Synthesize
    // extra wake-word clips for the SAME train voices at additional seeds and
    // run them through the same VAD-gated enrollment path.  The CLASSIFIER
    // keeps the frozen `pos_sequences` (its acceptance is already met and
    // untouched); only the verifier trains on the expanded pool.
    let extra_verifier_clips = generate_extra_verifier_positives_cached(
        &available_styles,
        &model_version_hash,
        &cache_dir_path,
    );
    info!(
        "Verifier positive expansion: {} extra train clips (mahbot-1025)",
        extra_verifier_clips.len(),
    );
    let extra_verifier_pos = vad_segment_and_enroll(&extra_verifier_clips);
    let mut verifier_pos_sequences: Vec<EmbeddingSequence> = pos_sequences.clone();
    let n_base_verifier_pos = verifier_pos_sequences.len();
    verifier_pos_sequences.extend(extra_verifier_pos);
    info!(
        "Verifier positive pool: {} base + {} extra = {} sequences (mahbot-1025)",
        n_base_verifier_pos,
        verifier_pos_sequences.len() - n_base_verifier_pos,
        verifier_pos_sequences.len(),
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
    // Build verifier negative sequences in 5-tier order:
    // ambient → owner → unrelated → confusable → cross-speaker (mahbot-1025).
    let mut verifier_neg_seqs: Vec<EmbeddingSequence> = Vec::new();
    verifier_neg_seqs.extend(ambient_sequences);
    verifier_neg_seqs.extend(owner_seqs);
    verifier_neg_seqs.extend_from_slice(unrelated_dense_cache);
    verifier_neg_seqs.extend_from_slice(confusable_dense_cache);
    // Cross-speaker TTS wake-word negatives (M2-M5 test clips, clean + 10/20 dB
    // white/pink) — in-distribution regression canaries under single-speaker
    // semantics (mahbot-1025).  Verifier-only: the classifier negative set is
    // untouched above (classifier_neg_seqs).
    let cross_speaker_seqs = generate_cross_speaker_negative_sequences(&test_clips);
    let n_cross_speaker = cross_speaker_seqs.len();
    info!(
        "Generated {n_cross_speaker} cross-speaker wake-word negative sequences \
         from {} held-out test clips (clean + 10/20 dB white/pink, mahbot-1025)",
        test_clips.len(),
    );
    verifier_neg_seqs.extend(cross_speaker_seqs);
    verifier_neg_sequences = verifier_neg_seqs;

    // Build per-sequence 5-tier weights matching production (mahbot-932 Fix 8)
    // plus the mahbot-1025 cross-speaker tier:
    // ambient (1.0×) → owner-negative (OWNER_NEGATIVE_UPWEIGHT×)
    // → unrelated (UNRELATED_UPWEIGHT×) → confusable (CONFUSABLE_UPWEIGHT×)
    // → cross-speaker (CROSS_SPEAKER_UPWEIGHT×).
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
    let n_seq_total = n_ambient + n_owner + n_confusable + n_unrelated + n_cross_speaker;

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
    pw.extend(std::iter::repeat_n(
        crate::audio::voice_verifier::CROSS_SPEAKER_UPWEIGHT,
        n_cross_speaker,
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
    crate::audio::voice_verifier::assert_weight_tier(
        &per_negative_sequence_weights,
        n_ambient + n_owner + n_unrelated + n_confusable,
        n_cross_speaker,
        crate::audio::voice_verifier::CROSS_SPEAKER_UPWEIGHT,
        "cross-speaker",
    );
    assert_eq!(per_negative_sequence_weights.len(), n_seq_total);

    info!(
        "Built {} verifier neg sequences ({} ambient@1.0× + {} owner@{}× + {} unrelated@{}× + {} confusable@{}× + {} cross-speaker@{}×) \
         and {} classifier neg sequences (mahbot-932 Fix 8; cross-speaker tier mahbot-1025)",
        n_seq_total,
        n_ambient,
        n_owner,
        crate::audio::voice_verifier::OWNER_NEGATIVE_UPWEIGHT,
        n_unrelated,
        crate::audio::voice_verifier::UNRELATED_UPWEIGHT,
        n_confusable,
        crate::audio::voice_verifier::CONFUSABLE_UPWEIGHT,
        n_cross_speaker,
        crate::audio::voice_verifier::CROSS_SPEAKER_UPWEIGHT,
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

    let (verifier, verifier_metrics) = VoiceVerifier::train_ensemble_with_metrics(
        &verifier_pos_sequences,
        &verifier_neg_sequences,
        Some(&per_negative_sequence_weights), // per-sequence weights matching production (mahbot-870 Fix 3)
        DEFAULT_VERIFIER_THRESHOLD,
        CONV_L2_LAMBDA, // Conv1D L2 regularization
                        // mahbot-1025: production's seed policy is preserved (entropy-based
                        // base seed per run — never seed-pinned) but the verifier is now a
                        // VERIFIER_ENSEMBLE_SEEDS-member multi-seed ensemble whose predict()
                        // averages all member scores, shrinking per-run seed variance.  The
                        // benchmark matches production exactly (both call
                        // train_ensemble_with_metrics).  The positive pool is the expanded
                        // verifier_pos_sequences (base + extra TTS seeds, mahbot-1025 item 2).
    );

    if verifier.is_trained() {
        info!(
            "VoiceVerifier trained successfully with {} dense positive + {} negative sequences \
             (ambient + owner-negative + confusable + unrelated + cross-speaker, mahbot-932/1025).  \
             Calibrated threshold={:.4}.",
            verifier_pos_sequences.len(),
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

    // mahbot-1012: same-audio training-vs-streaming comparison (Phase 8b).
    // Declared here so the JSON report can emit it even when the classifier
    // is degenerate (the report then carries an empty/partial section).
    let mut same_audio = Mahbot1012Report::default();
    // mahbot-1022: B-sweep measurement report (Phase 8c).  Declared here so
    // the JSON report can emit a note even when the classifier is degenerate
    // or the F1 clip is unavailable.
    let mut bsweep_report: Option<BsweepReport> = None;
    // mahbot-1023: enrolled-speaker benchmark phase (Phase 8d).  Declared
    // here so the JSON report can emit a note even when the classifier is
    // degenerate or the F1 clip is unavailable.
    let mut enrolled_report: Option<serde_json::Value> = None;

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
            "Cross-speaker detection (warm): {}/{} ({:.1}%) — M2–M5 clips are now \
             in-distribution negatives (mahbot-1025), so ~0 is the healthy reading; \
             the positive class is the enrolled speaker (Phase 8d)",
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
            "Cross-speaker detection (cold): {}/{} ({:.1}%) — M2–M5 clips are now \
             in-distribution negatives (mahbot-1025), so ~0 is the healthy reading",
            cold_metrics.detected,
            cold_metrics.total,
            cold_metrics.detection_rate() * 100.0,
        );
        phase_times[P_POSITIVE_VARIANTS] = phase_end_ms!();

        // ── Phase 8b: mahbot-1012 same-audio comparison ───────────────────
        // Runs INSIDE the Phase 8 block (before the negative phases so the
        // shared adaptive state is untouched) but is timed separately and
        // reported in the `mahbot1012` JSON section — no phase-numbering
        // ripple.  Measurement-only: the same PCM through the enrollment
        // path AND the streaming path, the anchor probe, the training-clips
        // control, and the exactly-one-of-five localization buckets.
        eprintln!("─── Phase 8b: mahbot-1012 same-audio comparison ───");
        info!("─── Phase 8b: mahbot-1012 same-audio comparison ───");
        same_audio = run_mahbot1012_comparison(
            &pos_test_variants,
            &train_clips,
            &classifier,
            &verifier,
            &pos_metrics.per_variant,
            &cold_metrics.per_variant,
        );
        // ── Phase 8c: B-sweep (mahbot-1022) — measurement only ────────────
        // Static-gate condition measurement over the streaming-layout mel
        // buffer length B (44..68) × in-buffer position (0/8/16/24) on the F1
        // training clip and its raw-level variants.  Runs AFTER Phase 8b so
        // the same-run F1 streaming trajectory (sanity anchor / cross-check /
        // trailing post-trim) is available; the sweep advances the shared VAD
        // detector (disclosed in the report — Phase 9+ negatives see drift).
        // Guarded against a degenerate classifier by the enclosing
        // `if !degenerate` block (line ~5979): when training produced an
        // all-zero classifier, `bsweep_report` stays `None` and the report
        // assembly emits the skip note instead.
        let bsweep_start = Instant::now();
        bsweep_report = run_bsweep(&train_clips, &classifier, &verifier, &same_audio);
        let bsweep_ms = bsweep_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  B-sweep completed in {bsweep_ms:.0}ms");

        // ── Phase 8d: Enrolled-speaker benchmark (mahbot-1023) ────────────
        // End-to-end detection of F1's 5 raw-level augmentation variants
        // through the REAL streaming cold pass (deferred burst + segment-end
        // pass + adaptive bootstrap + cold-start verifier warm-up).  Runs
        // AFTER Phase 8c so the same-run weights fingerprints are available
        // for interpretability; advances the shared VAD detector (disclosed
        // in the report — Phase 9+ negatives see the same drift as the
        // bsweep).  Labeled in-sample / training-side control, NOT a
        // generalization measure.  Measurement-only — the report is not
        // pass/fail gated (mahbot-953), but the acceptance protocol judges
        // the mean across ≥ 3 runs.
        eprintln!("─── Phase 8d: enrolled-speaker benchmark (mahbot-1023) ───");
        info!("─── Phase 8d: enrolled-speaker benchmark (mahbot-1023) ───");
        let enrolled_start = Instant::now();
        enrolled_report = run_enrolled_speaker_phase(&train_clips, &classifier, &verifier);
        let enrolled_ms = enrolled_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  Enrolled-speaker phase completed in {enrolled_ms:.0}ms");
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

    // Compute total false accepts across all categories.
    // mahbot-1025: the noise_overlap cross-speaker cells are now official FA
    // canaries — every detected variant across ALL noise_overlap cells is a
    // cross-speaker false accept (a non-enrolled speaker's wake word accepted
    // under noise; single-speaker semantics).  The per-cell `detail` JSON
    // carries per-variant `detected` flags, so we sum them.
    let noise_overlap_cross_speaker_fas: usize = noise_overlap_results
        .iter()
        .flat_map(|(_, _, detail)| detail.iter())
        .filter(|v| {
            v.get("detected")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        })
        .count();
    // The acceptance protocol's 0/300 target is 15 cells × ~20 variants —
    // derive the denominator from the actual measured cell×variant matrix so
    // every report string below stays in lockstep with the real FA count.
    let noise_overlap_total_variants: usize = noise_overlap_results
        .iter()
        .map(|(_, _, detail)| detail.len())
        .sum();
    let shared_fa_count = unrelated_metrics.false_accepts.len()
        + silence_metric.false_accepts.len()
        + noise_false_accepts.len()
        + noise_overlap_cross_speaker_fas;

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

    // Enrolled-speaker detection counts — the single-speaker POSITIVE class
    // (mahbot-1025 reclassification moved the M2–M5 cross-speaker clips to
    // in-distribution negatives, so Phase-8 pos_metrics is no longer a
    // positive signal).  Shared by the report banner, the threshold-status
    // block, and the catastrophic regression guard.  Falls back to Phase-8
    // counts when the enrolled phase did not run (degenerate classifier / F1
    // clip unavailable).
    let (enrolled_detected, enrolled_total): (usize, usize) =
        if let Some(acc) = enrolled_report.as_ref().and_then(|r| r.get("acceptance")) {
            let d = acc
                .get("detected_live")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as usize;
            let t = acc
                .get("total_variants")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(d as u64) as usize;
            (d, t)
        } else {
            (pos_metrics.detected, pos_metrics.total)
        };
    // Whether the enrolled-speaker phase (Phase 8d) actually ran.  When it did
    // not (degenerate classifier / F1 clip unavailable), enrolled_detected/
    // enrolled_total fall back to Phase-8 CROSS-SPEAKER counts — which are
    // negatives under the mahbot-1025 reclassification and therefore NOT a
    // positive detection signal.  The MIN_DETECTION_RATE suggestion and the
    // ratchet guard are skipped in that case so a degenerate run cannot print
    // a ratchet-to-zero 0.0 suggestion (mahbot-1008 ratchet guard).
    let enrolled_phase_ran = enrolled_report
        .as_ref()
        .and_then(|r| r.get("acceptance"))
        .is_some();

    // ── Verifier recall on wake-word variants (mahbot-1008 Fix 6) ──
    // The pre-fix verifier accepted 0/17 positive variants (constant 6.67e-8
    // reject-all) while the benchmark reported the 0% detection rate without a
    // verifier-recall signal.  Compute the accept rate over variants the
    // verifier actually evaluated.  mahbot-1025: the positive class is now the
    // ENROLLED speaker's wake word (Phase 8d) — the Phase-8 M2–M5 cross-speaker
    // clips were reclassified as in-distribution negatives the verifier must
    // REJECT, so their ~0 accept rate is correct, not a recall failure.
    // Computed before the report so both the human-readable report and the JSON
    // output use the same value.
    let verifier_recall_rate = if let Some(arr) = enrolled_report
        .as_ref()
        .and_then(|r| r.get("per_variant"))
        .and_then(serde_json::Value::as_array)
        .filter(|arr| !arr.is_empty())
    {
        // Enrolled-speaker F1 variants: the verifier is evaluated on every
        // variant; accepted = verifier peak >= the constant 0.86 acceptance
        // floor.
        let evaluated = arr.len();
        let accepted = arr
            .iter()
            .filter(|v| {
                v.get("verifier_peak")
                    .and_then(serde_json::Value::as_f64)
                    .is_some_and(|p| {
                        p >= f64::from(crate::audio::voice_verifier::VERIFIER_ACCEPTANCE_FLOOR)
                    })
            })
            .count();
        if evaluated == 0 {
            None
        } else {
            Some((accepted, evaluated))
        }
    } else {
        verifier_recall(&pos_metrics.per_variant)
    };

    info!("══════════════════════════════════════════════");
    info!("      Voice Pipeline E2E Benchmark Results");
    info!("══════════════════════════════════════════════");
    info!(
        "Total benchmark time: {:.1}s",
        overall_start.elapsed().as_secs_f64()
    );
    info!(
        "Enrolled-speaker detection rate: {:.1}% ({}/{}) — target ≥{:.0}% \
         (single-speaker positive class; Phase-8 cross-speaker clips are \
         negatives by design, mahbot-1025)",
        if enrolled_total > 0 {
            enrolled_detected as f64 / enrolled_total as f64 * 100.0
        } else {
            0.0
        },
        enrolled_detected,
        enrolled_total,
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
    // mahbot-1025: the Phase-8 "positive" variants are the M2–M5 cross-speaker
    // clips, reclassified as in-distribution negatives the verifier rejects —
    // they no longer represent the positive class for calibration.  The sweep
    // uses the enrolled-speaker F1 variant verifier peaks from Phase 8d (the
    // true single-speaker positive class); when the enrolled phase did not run
    // (degenerate classifier / F1 clip unavailable) it falls back to the
    // Phase-8 peaks (degraded — cross-speaker rejection readings).
    let pos_verifier_scores: Vec<f32> = enrolled_report
        .as_ref()
        .and_then(|r| r.get("per_variant"))
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.get("verifier_peak"))
                .filter_map(serde_json::Value::as_f64)
                .map(|p| p as f32)
                .collect::<Vec<f32>>()
        })
        .filter(|scores: &Vec<f32>| !scores.is_empty())
        .unwrap_or_else(|| {
            pos_metrics
                .per_variant
                .iter()
                .map(|pv| pv.verifier_score)
                .collect()
        });

    let n_pos_total = pos_verifier_scores.len();
    let benchmark_detection_rate = if enrolled_total > 0 {
        // Enrolled-speaker end-to-end detection rate (the true positive class,
        // mahbot-1025 reclassification).
        enrolled_detected as f64 / enrolled_total as f64
    } else if n_pos_total > 0 {
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
    // mahbot-1025: the rate is the ENROLLED-SPEAKER end-to-end detection rate
    // (5 F1 variants — each miss costs 20 points) because the Phase-8 M2–M5
    // cross-speaker clips were reclassified as negatives; the suggestion is
    // report-only and the ratchet guard below never lowers the constant on a
    // failing run.
    let computed_min_dr: Option<f64> = if enrolled_phase_ran && enrolled_total > 0 {
        // The enrolled-speaker detection rate is the true single-speaker
        // positive signal (mahbot-1025 reclassification); the suggestion is
        // only meaningful when that phase actually ran.  Add epsilon before
        // floor() to guard against IEEE 754 imprecision: e.g., 0.80 / 0.2 =
        // 3.999999... would floor to 3 without epsilon.
        let dr_f64 = enrolled_detected as f64 / enrolled_total as f64;
        Some(((dr_f64 / 0.2 + 1e-12).floor() * 0.2 * 1000.0).round() / 1000.0)
    } else {
        None
    };

    let threshold_divergence = (benchmark_dual_threshold - verifier_training_threshold).abs();

    info!(
        "Verifier threshold analysis (mahbot-997): \
         training_threshold={verifier_training_threshold:.4}, \
         benchmark-optimal threshold (post-hoc dual sweep)={benchmark_dual_threshold:.4}, \
         divergence={threshold_divergence:.4}",
    );
    if let Some(computed_min_dr) = computed_min_dr {
        info!(
            "MIN_DETECTION_RATE hint: current_constant={MIN_DETECTION_RATE:.2}, \
             computed={computed_min_dr:.3} (= floor({:.3} / 0.2) × 0.2).  \
             Update MIN_DETECTION_RATE constant if this run is representative.",
            benchmark_detection_rate,
        );
    } else {
        info!(
            "MIN_DETECTION_RATE hint skipped: enrolled-speaker phase did not run — \
             no positive detection signal to suggest a rate from (mahbot-1025).",
        );
    }
    // ── Ratchet guard (mahbot-1008) ──
    // The mahbot-997 auto-suggestion formula floors to 0.0 on failing runs,
    // which would endorse 0% detection as passing by ratcheting the constant
    // toward zero.  A failing run must never lower the bar.  Guarded on the
    // enrolled phase actually running: when it did not, benchmark_detection_rate
    // is the cross-speaker fallback (negatives, ~0), which is not a meaningful
    // positive rate and must not trigger the warning.
    if enrolled_phase_ran && benchmark_detection_rate < MIN_DETECTION_RATE {
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

    // Noise-overlap: under the mahbot-1025 reclassification the noise_overlap
    // cells are CROSS-SPEAKER NEGATIVES (non-enrolled speakers' wake words,
    // single-speaker semantics) — any detection is a cross-speaker FALSE
    // ACCEPT, and the acceptance protocol requires 0/{noise_overlap_total_variants}
    // per fresh run.  The per-cell rate is reported and the cross-speaker FA
    // count joins the official FA tally (`noise_overlap_cross_speaker_fas`
    // above).  Warnings only — report-only (mahbot-953).
    for (key, rate, detail) in &noise_overlap_results {
        let cell_fas = detail
            .iter()
            .filter(|v| {
                v.get("detected")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            })
            .count();
        if cell_fas > 0 {
            warn!(
                "Noise-overlap cross-speaker FA: {key} accepted {cell_fas}/{detail_len} cells \
                 (rate {rate_pct:.1}%) — target 0/{noise_overlap_total_variants} per fresh run (mahbot-1025)",
                detail_len = detail.len(),
                rate_pct = rate * 100.0,
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
        .map(|(pv, cat)| pv_to_json(pv, Some(cat), None))
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
            // mahbot-1025: the verifier is a multi-seed ensemble — one entropy
            // base seed per run (never pinned) with VERIFIER_ENSEMBLE_SEEDS
            // member seeds derived deterministically from it; predict() = member
            // mean.  The single-seed `None` policy (mahbot-1006 K) is preserved
            // at the base-seed level; the ensemble shrinks the per-run variance
            // that the old single draw exposed.
            "verifier_ensemble_members": crate::audio::voice_verifier::VERIFIER_ENSEMBLE_SEEDS,
            "verifier_base_seed": "entropy (per run, never pinned)",
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
        "mahbot1012_note": "Same-audio training-vs-streaming comparison (Phase 8b): \
                            per-frame embedding hashes / window geometry / mel buffer \
                            length / VAD hops / adaptive-mode / candidate trajectory in \
                            every positive per_variant result; dual-path same-audio \
                            capture for all positive variants AND all training clips; \
                            raw-mel comparison for one representative variant; \
                            anchor-comparison probe for the first variant of each \
                            augmentation type.  All report-only, gated behind \
                            voice-tests.  The verifier is a multi-seed ensemble (mahbot-1025) — the \
                            localization bucket counts carry run-to-run variance, while \
                            the hash/cosine/L2 evidence is seed-independent.  This phase \
                            adds ~2-5 min to a cached run; on a cold TTS cache the total \
                            may approach the 30-minute harness timeout — progress is \
                            printed to stderr per variant.",
    });

    // ── JSON sub-objects (built separately to stay under serde_json's json!
    // macro recursion limit — the main report object nests several levels) ──

    // Detection summary: warm (pos_metrics) + cold-start (cold_metrics, mahbot-1006 D).
    // mahbot-1025: the Phase-8 "positive" variants are the M2–M5 CROSS-SPEAKER
    // clips, reclassified as in-distribution negatives the verifier rejects —
    // a healthy pipeline's warm/cold detection rate here is ~0 BY DESIGN.  The
    // positive class under single-speaker semantics is the enrolled speaker's
    // wake word, reported in the enrolled_speaker.acceptance section
    // (detected_live); the F1 TP acceptance gate reads that section.
    let detection_json = serde_json::json!({
        "rate": if pos_metrics.total > 0 {
            serde_json::Value::from(pos_metrics.detection_rate())
        } else {
            serde_json::Value::Null
        },
        "detected": pos_metrics.detected,
        "total_positive": pos_metrics.total,
        "no_tests_ran": pos_metrics.total == 0,
        "note": "These are the M2–M5 CROSS-SPEAKER clips (mahbot-1025 reclassification): \
                 in-distribution negatives the verifier is trained to reject, so ~0 \
                 detection is the healthy reading.  The single-speaker positive class \
                 is the enrolled speaker's wake word — see enrolled_speaker.acceptance.",
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
                // mahbot-1025: noise_overlap cells are cross-speaker negatives —
                // detection = cross-speaker FA (official tally).
                let cross_speaker_fas = detail
                    .iter()
                    .filter(|v| {
                        v.get("detected")
                            .and_then(serde_json::Value::as_bool)
                            .unwrap_or(false)
                    })
                    .count();
                (
                    key.clone(),
                    serde_json::json!({
                        "detection_rate": rate,
                        "cross_speaker_fas": cross_speaker_fas,
                        "note": format!(
                            "cross_speaker_fas = accepted non-enrolled-speaker cells \
                             (official FA canaries, mahbot-1025; target 0/{noise_overlap_total_variants} per run)",
                        ),
                        "per_variant": detail,
                    }),
                )
            })
            .collect(),
    );

    // mahbot-1022: B-sweep section merged with the same-run negatives extract.
    // Filled from the Phase 8c report when the classifier is non-degenerate
    // and the F1 clip is present; otherwise a note explains the skip.
    let bsweep_json = bsweep_report.as_ref().map_or_else(
        || {
            serde_json::json!({"note": "B-sweep skipped (degenerate classifier or F1 clip unavailable)"})
        },
        |r| r.to_json(negative_verifier_extraction(&all_neg_pv)),
    );

    // mahbot-1023: enrolled-speaker phase (Phase 8d) + safety gate (item 7).
    // The safety gate measures the false-positive impact of the deferred
    // in-distribution windows (burst + boundary double-scoring) on the
    // negative/confusable set; baseline 9/59 gate crossings / 0 FAs.
    let enrolled_json = enrolled_report.unwrap_or_else(|| {
        serde_json::json!({"note": "Enrolled-speaker phase skipped (degenerate classifier or F1 clip unavailable)"})
    });
    // mahbot-1025: hoist the per-member speed_down peaks from the enrolled
    // report into the top-level verifier_diagnostics (the variance-reduction
    // gate).  Defaults: empty peaks / miss fraction 0 when the phase was
    // skipped.  The miss fraction is read from the enrolled report itself
    // (single source of truth — it already excludes untrained members and
    // fails closed at 1.0 when no trained member was measurable), not
    // recomputed here.
    let (speed_down_member_peaks, speed_down_member_miss_fraction): (Vec<f32>, f32) = enrolled_json
        .get("acceptance")
        .and_then(|a| a.get("speed_down_member_peaks"))
        .and_then(|p| p.as_array())
        .map(|arr| {
            let peaks: Vec<f32> = arr
                .iter()
                .filter_map(|v| v.as_f64().map(|x| x as f32))
                .collect();
            let miss = enrolled_json
                .get("acceptance")
                .and_then(|a| a.get("speed_down_member_miss_fraction"))
                .and_then(serde_json::Value::as_f64)
                .map_or(0.0, |m| m as f32);
            (peaks, miss)
        })
        .unwrap_or_default();
    let safety_gate_json = safety_gate(&all_neg_pv);

    let json = serde_json::json!({
        "benchmark": "voice_pipeline_e2e",
        "_note": format!(
            "Report-only benchmark — no pass/fail gating (mahbot-953).  \
             'passed' was removed in mahbot-1005: it was hardcoded true and \
             misleading.  Compare against the limits in 'results' instead. \
             mahbot-1006 aligned the benchmark's training/inference \
             processing with production (AGC→augment ordering, cold-start \
             pass, CONFIG-driven preprocessor, preprocessed negatives, \
             verifier seed None) — detection/FA numbers are NOT directly \
             comparable to the mahbot-1004/948 baselines.  mahbot-1025: \
             the verifier is now a {}-member multi-seed ensemble (entropy \
             base seed per run, member seeds derived deterministically); \
             cross-speaker M2-M5 wake-word negatives (clean + 10/20 dB \
             white/pink) joined the verifier negative set and the \
             noise_overlap cross-speaker cells are official FA canaries.",
            crate::audio::voice_verifier::VERIFIER_ENSEMBLE_SEEDS,
        ),
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
            // mahbot-1025: official cross-speaker FA canaries — any detection
            // of a non-enrolled speaker's wake word in the noise_overlap cells.
            "noise_overlap_cross_speaker": noise_overlap_cross_speaker_fas,
            "noise_overlap_cross_speaker_target": format!(
                "0/{noise_overlap_total_variants} per fresh run (across all three runs)"
            ),
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
            pos_metrics
                .per_variant
                .iter()
                .map(|pv| pv_to_json(pv, None, same_audio.row_for(&pv.variant, "warm")))
                .collect()
        ),
        // mahbot-1006 D: cold-start pass per-variant detail (same schema as
        // per_variant_results, but measured without consume_warmup / fresh
        // adaptive bootstrap).  Empty when the classifier is degenerate.
        // mahbot-1023: measured under the deferred-burst pipeline — the cold
        // pass exercises the burst sweep (buffer >= 68 frames) or the
        // segment-end pass for short utterances, so these numbers reflect
        // the fix, not the mahbot-1022 0/20 baseline.  The enrolled-speaker
        // phase (enrolled_speaker section) is the acceptance-relevant
        // in-sample measurement; this section is the same pipeline over the
        // full 20-variant positive test set.
        "per_variant_results_cold": serde_json::Value::Array(
            cold_metrics
                .per_variant
                .iter()
                .map(|pv| pv_to_json(pv, None, same_audio.row_for(&pv.variant, "cold")))
                .collect()
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
            // mahbot-1025: multi-seed ensemble size (1 = legacy single member).
            "ensemble_size": verifier.ensemble_size(),
            "ensemble_seeds": crate::audio::voice_verifier::VERIFIER_ENSEMBLE_SEEDS,
            // mahbot-1025: per-member speed_down peak distribution (the
            // variance-reduction gate).  Computed on the F1 speed_down variant
            // through each member's standalone verifier; the fraction of
            // members below VERIFIER_ACCEPTANCE_FLOOR is the per-run
            // single-seed miss-probability estimate (target ≤0.15, band
            // ~0.04–0.10).
            "speed_down_member_peaks": speed_down_member_peaks,
            "speed_down_member_miss_fraction": speed_down_member_miss_fraction,
            // mahbot-1023: the acceptance gate is the constant 0.86 floor —
            // `threshold` above is a report-only reference.
            "effective_acceptance_threshold": crate::audio::voice_verifier::VERIFIER_ACCEPTANCE_FLOOR,
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
            // mahbot-1023: the runtime-calibrated / fallback threshold is a
            // report-only reference — the confirmation gate uses the CONSTANT
            // VERIFIER_ACCEPTANCE_FLOOR (0.86) in all cases (manager pin 1).
            "effective_acceptance_threshold": crate::audio::voice_verifier::VERIFIER_ACCEPTANCE_FLOOR,
            "note": "Effective acceptance threshold is the constant 0.86 \
                     (VERIFIER_ACCEPTANCE_FLOOR) — the user-approved production \
                     semantics — NOT the entropy-seeded runtime-calibrated value \
                     reported above (which drifts across runs: 0.86 -> 0.91 on \
                     two identical-code runs) and NOT the DEFAULT_VERIFIER_THRESHOLD \
                     0.948 fallback.  The runtime values remain report-only so \
                     threshold drift is reviewable over the >=3-run protocol.",
            "benchmark_optimal_threshold_dual": benchmark_dual_threshold,
            "dual_sweep_detection_rate_at_optimal": dual_sweep_dr,
            "dual_sweep_false_accept_rate_at_optimal": dual_sweep_far,
            "threshold_divergence": threshold_divergence,
            "computed_min_detection_rate_suggestion": computed_min_dr,
            "computed_min_detection_rate_note": match computed_min_dr {
                Some(_) => format!(
                    "= floor({benchmark_detection_rate:.3} / 0.2) × 0.2, from the \
                     enrolled-speaker detection rate",
                ),
                None => "skipped — enrolled-speaker phase did not run (degenerate classifier \
                         / F1 clip unavailable); the cross-speaker fallback is not a positive \
                         signal (mahbot-1025)"
                    .to_string(),
            },
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
        "mahbot1012": same_audio.to_json(),
        "bsweep": bsweep_json,
        // mahbot-1023: enrolled-speaker benchmark phase (Phase 8d) — F1's 5
        // raw-level augmentation variants through the real streaming cold
        // pass (in-sample / training-side control, NOT generalization).
        "enrolled_speaker": enrolled_json,
        // mahbot-1023 item 7: false-positive impact of the deferred
        // in-distribution windows on the negative/confusable set.
        "safety_gate": safety_gate_json,
        "config": {
            "num_enrollment_variants": NUM_ENROLLMENT_VARIANTS,
            "min_detection_rate": MIN_DETECTION_RATE,
            "verifier_recall_min": VERIFIER_RECALL_MIN,
        }
    });

    // Output delimited JSON for CI tooling
    println!("--- BENCHMARK_JSON_BEGIN ---");
    let json_text = serde_json::to_string_pretty(&json).expect("JSON serialization");
    println!("{json_text}");
    println!("--- BENCHMARK_JSON_END ---");

    // ── Persist the report for post-run analysis (mahbot-1012 reviewer) ────
    // The report is emitted to stdout for CI, but a hard 30-minute harness
    // timeout with end-only emission can lose it entirely.  Mirror the JSON to
    // ~/.mahbot/voice_pipeline_e2e_report.json (the same directory as the
    // benchmark lock file) so a run's data survives regardless of how the
    // process ends, and a final ticket comment can cite exact numbers.
    if let Ok(report_dir) = crate::config::default_config_dir() {
        let report_path = report_dir.join("voice_pipeline_e2e_report.json");
        // ── Per-run timestamped archive (mahbot-1023, manager pin 5) ──────
        // The ≥3-run acceptance protocol computes spread from ACTUAL runs,
        // which the old single .prior.json scheme cannot support (a 3rd run
        // would overwrite run 1's archive).  Every run is archived with a
        // UTC timestamp before the main report path is overwritten, so the
        // acceptance review has all runs' reports.  The .prior.json copy is
        // retained for backward compatibility with existing tooling.
        if report_path.exists() {
            let prior_path = report_dir.join("voice_pipeline_e2e_report.prior.json");
            if let Err(e) = std::fs::copy(&report_path, &prior_path) {
                warn!(
                    "Could not preserve prior benchmark report to {}: {e}",
                    prior_path.display()
                );
            }
        }
        let archive_stamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
        let archive_path =
            report_dir.join(format!("voice_pipeline_e2e_report.{archive_stamp}.json"));
        match std::fs::write(&archive_path, &json_text) {
            Ok(()) => info!(
                "Benchmark report archived to {} (acceptance spread computed from timestamped per-run archives)",
                archive_path.display()
            ),
            Err(e) => warn!(
                "Could not write benchmark report archive to {}: {e}",
                archive_path.display()
            ),
        }
        match std::fs::write(&report_path, json_text) {
            Ok(()) => info!("Benchmark report written to {}", report_path.display()),
            Err(e) => warn!(
                "Could not write benchmark report to {}: {e}",
                report_path.display()
            ),
        }
    } else {
        warn!("Could not resolve ~/.mahbot/ for benchmark report persistence");
    }

    // ── Human-readable report (stderr, mahbot-871) ────────────────────────
    let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    // mahbot-1025: the headline detection rate is the ENROLLED-speaker rate
    // (the single-speaker positive class); the Phase-8 M2–M5 cross-speaker
    // clips were reclassified as in-distribution negatives the verifier
    // rejects (their ~0 detection is correct, not a regression).
    let dr = if enrolled_total > 0 {
        enrolled_detected as f64 / enrolled_total as f64
    } else {
        0.0
    };
    // NOTE: checkmarks are informational only — benchmark is report-only, no
    // pass/fail gating (mahbot-953).  The marks now reflect the ACTUAL limits
    // instead of being unconditional ✓ (mahbot-1005 §7).
    let mk = |pass: bool| if pass { '✓' } else { '✗' };
    let dr_pass = if enrolled_total > 0 && dr >= MIN_DETECTION_RATE {
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
    // MIN_DR suggestion line: null (None) when the enrolled phase did not run
    // — the cross-speaker fallback is not a positive signal to suggest from.
    // Note: fractions are formatted as percentages (×100) — the banner is the
    // human-readable acceptance report and must match the JSON values.
    let min_dr_suggestion = computed_min_dr.map_or_else(
        || "skipped (enrolled phase did not run)".to_string(),
        |v| format!("{:.0}%", v * 100.0),
    );

    eprintln!(
        "\n\
         ═══════════════════════════════════════════════════════════\n\
                 Voice Pipeline E2E Benchmark Report\n\
         ═══════════════════════════════════════════════════════════\n\
         Date/Time:      {timestamp}\n\
         Tier:           Easy/Medium/Hard (per-tier limits)\n\
         Detection rate: {dr_pct:.1}% ({enrolled_detected}/{enrolled_total})  {dr_pass}\n\
         MIN_DR target:  ≥{min_dr_const_pct:.0}% (suggestion: {min_dr_suggestion})\n\
         Threshold:      training={verifier_training_threshold:.4}, dual-sweep-optimal={benchmark_dual_threshold:.4}\n\
         False accepts:\n\
           Confusable:\n\
             Easy:    {fa_easy}",
        dr_pct = dr * 100.0,
        min_dr_const_pct = MIN_DETECTION_RATE * 100.0,
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
        "           Noise-overlap (cross-speaker): {cross_count}  {cross_mark} (target 0/{cross_target} per fresh run)",
        cross_count = noise_overlap_cross_speaker_fas,
        cross_mark = mk(noise_overlap_cross_speaker_fas == 0),
        cross_target = noise_overlap_total_variants,
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

    // Catastrophic regression guard: at least 1 positive detection must occur.
    // Without this, a pipeline that detects nothing would pass all FA assertions
    // (zero detections = zero false accepts) (mahbot-911).
    // NOTE: report-only — warns instead of asserting (mahbot-953).
    // mahbot-1025: the Phase-8 "positive" variants are the M2–M5 CROSS-SPEAKER
    // clips, reclassified as in-distribution negatives the verifier is trained
    // to reject — a healthy pipeline detects ~0 of them, so pos_metrics no
    // longer measures the positive class.  The positive class under
    // single-speaker semantics is the enrolled speaker's wake word, measured
    // end-to-end by the Phase-8d enrolled-speaker phase (detected_live).
    let (positive_detected, positive_total): (u64, u64) =
        (enrolled_detected as u64, enrolled_total as u64);
    if positive_detected == 0 {
        warn!(
            "Catastrophic regression: 0/{positive_total} enrolled-speaker wake word variants \
             detected — pipeline is not detecting the enrolled speaker's wake word.  \
             Detection rate: {dr:.1}%",
            dr = if positive_total > 0 {
                positive_detected as f64 / positive_total as f64 * 100.0
            } else {
                0.0
            },
        );
    }

    // Calibrated threshold check (mahbot-997).
    // The verifier threshold is now auto-calibrated during training.  This
    // informational section displays the benchmark-optimal threshold and the
    // computed MIN_DETECTION_RATE suggestion (see dual-sweep analysis above).
    // mahbot-1025: the "benchmark detection rate" is the enrolled-speaker
    // end-to-end detection rate (the true single-speaker positive class); the
    // Phase-8 M2–M5 cross-speaker clips were reclassified as in-distribution
    // negatives the verifier rejects, so their ~0 detection is correct and is
    // NOT a positive signal.
    let actual_dr = if enrolled_total > 0 {
        enrolled_detected as f64 / enrolled_total as f64
    } else {
        pos_metrics.detection_rate()
    };
    let detected_total = (enrolled_detected, enrolled_total);
    if enrolled_phase_ran {
        info!(
            "Verifier threshold status: trained={} auto-calibrated={:.4}, \
             enrolled-speaker detection rate={:.1}% ({}/{}), \
             benchmark-optimal (dual sweep)={benchmark_dual_threshold:.4}, \
             computed MIN_DETECTION_RATE suggestion={}",
            verifier.is_trained(),
            verifier_training_threshold,
            actual_dr * 100.0,
            detected_total.0,
            detected_total.1,
            computed_min_dr.map_or_else(
                || "skipped (enrolled phase did not run)".to_string(),
                |v| format!("{v:.3}"),
            ),
        );
    }

    // ── mahbot-1025 ensemble + cross-speaker summary ────────────────────
    eprintln!(
        "         Verifier ensemble: {n_members} members (seed ensemble), \
         speed_down member miss fraction: {miss_fraction:.3} (gate ≤0.15, target ~0.04–0.10), \
         cross-speaker noise_overlap FAs: {cross_fas}/{noise_overlap_total_variants} target",
        n_members = verifier.ensemble_size(),
        miss_fraction = speed_down_member_miss_fraction,
        cross_fas = noise_overlap_cross_speaker_fas,
    );
    if let Some(f1_noise_peak) = enrolled_json
        .get("acceptance")
        .and_then(|a| a.get("f1_noise_verifier_peak"))
        .and_then(serde_json::Value::as_f64)
    {
        eprintln!("         F1_noise verifier peak: {f1_noise_peak:.4} (must be ≥ 0.86)");
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
        per_frame_embedding_hashes: Vec::new(),
        per_frame_embedding_l2_norms: Vec::new(),
        per_frame_window_start: Vec::new(),
        per_frame_mel_buffer_len: Vec::new(),
        per_frame_geometry: Vec::new(),
        per_frame_adaptive_mode: Vec::new(),
        per_frame_candidate_state: Vec::new(),
        per_hop_vad: Vec::new(),
        // mahbot-1023: new fields default to the no-burst / no-detection /
        // not-persisted-bootstrap states.  The acceptance threshold is the
        // constant 0.86 floor (referenced directly at the gates, not stored
        // as a field — it can never vary).
        detection_path: None,
        candidate_created_path: None,
        burst_sweep_fired: false,
        burst_sweep_buffer_len: None,
        segment_end_pass_fired: false,
        adaptive_bootstrap_persisted: false,
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
    // 4. Warm-up epoch trigger (frame-accurate, mahbot-1024): the first
    //    threshold crossing fell INSIDE the verifier warm-up epoch (cold
    //    pass: warmup_n_embeddings=0, frame 1 < 3 warm-up frames) →
    //    warmup_suppression, even though the pre-utterance warmup_completed
    //    flag is false.
    {
        let mut pv = verdict_test_pv(false, 10, 5, false, None, 2.5, true, 0.0, true, 0.6);
        pv.warmup_n_embeddings = 0;
        pv.first_trigger_frame_idx = Some(1);
        assert_eq!(classify_miss(&pv), V::WarmupSuppression);
    }
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
    // 9. Verifier peak crossed the CONSTANT acceptance floor (0.86) but
    //    detection didn't fire → verifier_timing (candidate expired or audio
    //    ended first).  The runtime-calibrated threshold (0.6 here) is a
    //    report-only reference — the split uses VERIFIER_ACCEPTANCE_FLOOR
    //    (mahbot-1023).
    assert_eq!(
        classify_miss(&verdict_test_pv(
            false, 10, 5, true, None, 2.5, true, 0.9, true, 0.6
        )),
        V::VerifierTiming,
    );
    // 10. Verifier peak below the acceptance floor → verifier_rejected
    //     (0.3 < 0.86, even though it exceeds the runtime threshold 0.6).
    assert_eq!(
        classify_miss(&verdict_test_pv(
            false, 10, 5, true, None, 2.5, true, 0.3, true, 0.6
        )),
        V::VerifierRejected,
    );
    // 11. REGRESSION (mahbot-1024): the cold-pass speed_down miss must NOT
    //     be mislabeled as warmup_suppression.  The first trigger was frame
    //     3 — the exact warm-up boundary (VERIFIER_WARMUP_EMBEDDINGS = 4,
    //     warmup_n_embeddings = 0 → warm-up frames 0..2) — and the verifier
    //     genuinely rejected the candidate (peak 0.7886 < 0.86 floor) →
    //     verifier_rejected.  This pins the observed run 20260801-061348
    //     mislabeling.
    {
        let mut pv = verdict_test_pv(false, 10, 5, false, None, 2.5, true, 0.7886, true, 0.6);
        pv.warmup_n_embeddings = 0;
        pv.first_trigger_frame_idx = Some(3);
        assert_eq!(classify_miss(&pv), V::VerifierRejected);
    }
    // 12. Post-warm-up trigger with the verifier peak ABOVE the floor but no
    //     detection → verifier_timing (candidate lifecycle / audio ended),
    //     NOT warm-up suppression.
    {
        let mut pv = verdict_test_pv(false, 10, 5, false, None, 2.5, true, 0.9, true, 0.6);
        pv.warmup_n_embeddings = 0;
        pv.first_trigger_frame_idx = Some(4);
        assert_eq!(classify_miss(&pv), V::VerifierTiming);
    }
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

/// The enrolled-phase mechanism taxonomy (mahbot-1024 re-scope): a
/// main-loop ("other") detection of a burst-created candidate is the
/// PRIMARY mechanism ("burst_continuation"), NOT a regression; only a
/// main-loop detection whose candidate the main loop created itself is
/// "unexpected".  "burst" / "segment_end_pass" pass through unchanged.
#[test]
fn reclassify_detection_path_taxonomy() {
    // Burst sweep confirmed directly.
    assert_eq!(
        reclassify_detection_path(Some("burst"), Some("burst")),
        Some("burst".to_string()),
    );
    // Segment-end pass confirmed (candidate created by burst or main loop —
    // the boundary pass owns the attribution).
    assert_eq!(
        reclassify_detection_path(Some("segment_end_pass"), Some("burst")),
        Some("segment_end_pass".to_string()),
    );
    assert_eq!(
        reclassify_detection_path(Some("segment_end_pass"), Some("other")),
        Some("segment_end_pass".to_string()),
    );
    // PRIMARY mechanism: burst-created candidate confirmed by main-loop
    // continuation (the 12/14 archived final-code detections).
    assert_eq!(
        reclassify_detection_path(Some("other"), Some("burst")),
        Some("burst_continuation".to_string()),
    );
    // Genuine unexpected path: main loop created AND confirmed its own
    // candidate (no burst-created candidate to continue).
    assert_eq!(
        reclassify_detection_path(Some("other"), Some("other")),
        Some("unexpected".to_string()),
    );
    // Candidate origin unknown (no candidate tracking evidence) — treated as
    // unexpected rather than silently accepted.
    assert_eq!(
        reclassify_detection_path(Some("other"), None),
        Some("unexpected".to_string()),
    );
    // Not detected → None.
    assert_eq!(reclassify_detection_path(None, None), None);
    assert_eq!(reclassify_detection_path(None, Some("burst")), None);
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

// ── mahbot-1012 localization bucket unit tests ─────────────────────────────
// These pin the exactly-one-of-five bucket assignment and its quantitative
// boundaries (the (c) vs (d) boundary, the (b) hash-mismatch rule, the AGC
// correlated-diagnostic annotation, and the precedence chain).

/// Build a minimal `DualPathComparison` for bucket tests: `n` ordinal windows,
/// the first `n_match` of which hash-match the training path.
#[cfg(test)]
fn test_comparison(n_windows: usize, n_hash_match: usize) -> DualPathComparison {
    let mk_side = |n: usize, hash_base: u64| DualPathSide {
        window_start: (0..n).map(|i| i * 8).collect(),
        real_frames: vec![super::EMBEDDING_WINDOW_FRAMES; n],
        hashes: (0..n).map(|i| hash_base + i as u64).collect(),
        embedding_l2: vec![1.0; n],
        classifier_scores: vec![0.9; n],
        verifier_scores: vec![0.0; n],
        max_rolling_sum: 0.0,
        geometry: vec!["true_sliding"; n],
    };
    let streaming = mk_side(n_windows, 100);
    let training = mk_side(n_windows, 100);
    // Overwrite the first n_hash_match training hashes to equal the streaming
    // hashes (the rest differ).
    let mut training = training;
    for i in 0..n_hash_match {
        training.hashes[i] = streaming.hashes[i];
    }
    let matches: Vec<DualPathWindowMatch> = (0..n_windows)
        .map(|i| {
            let matched = i < n_hash_match;
            DualPathWindowMatch {
                streaming_idx: i,
                training_idx: Some(i),
                streaming_start: i * 8,
                training_start: Some(i * 8),
                start_delta: Some(0),
                streaming_real_frames: super::EMBEDDING_WINDOW_FRAMES,
                training_real_frames: super::EMBEDDING_WINDOW_FRAMES,
                hash_match: Some(matched),
                cosine: if matched { Some(1.0) } else { Some(0.0) },
                l2_delta: if matched { Some(0.0) } else { Some(1.0) },
                streaming_score: 0.9,
                training_score: Some(0.9),
                streaming_geometry: "true_sliding",
            }
        })
        .collect();
    DualPathComparison {
        label: "test".to_string(),
        detected: false,
        streaming,
        training,
        matches,
        first_mismatch_idx: if n_hash_match < n_windows {
            Some(n_hash_match)
        } else {
            None
        },
        hash_match_frac: if n_windows > 0 {
            n_hash_match as f64 / n_windows as f64
        } else {
            0.0
        },
        window0_cosine: Some(if n_hash_match > 0 { 1.0 } else { 0.0 }),
        window0_l2_delta: Some(if n_hash_match > 0 { 0.0 } else { 1.0 }),
    }
}

/// (a): zero scored windows → embeddings never reached the classifier.
#[test]
fn localization_bucket_a_embeddings_never_reached_classifier() {
    let pv = verdict_test_pv(false, 0, 0, true, None, 0.0, false, 0.0, true, 0.6);
    // Precedence: even a hash-mismatching comparison cannot override (a).
    let cmp = test_comparison(3, 0);
    let (bucket, _ev) = classify_localization_bucket(&pv, &cmp, Some(0.9));
    assert_eq!(bucket, LocalizationBucket::EmbeddingsNeverReachedClassifier);
}

/// (b): window-0 hash mismatch → embeddings differ (the mel values diverged
/// before any windowing/grid effect — the per-call mel floor signature).
#[test]
fn localization_bucket_b_window0_mismatch_is_embeddings_differ() {
    let pv = verdict_test_pv(false, 10, 3, true, None, 2.5, true, 0.6, true, 0.6);
    let cmp = test_comparison(3, 0); // first (window 0) mismatches, <50% match
    let (bucket, ev) = classify_localization_bucket(&pv, &cmp, Some(0.9));
    assert_eq!(bucket, LocalizationBucket::EmbeddingsDiffer);
    let reason = ev.get("reason").expect("reason present");
    assert!(
        reason.as_str().unwrap().contains("first ordinal window"),
        "reason should cite the window-0 mismatch rule, got: {reason}"
    );
}

/// (b): majority mismatch with a later first-mismatch is still embeddings
/// differ (grid drift / padded-window geometry divergence starts at window k).
#[test]
fn localization_bucket_b_late_majority_mismatch_is_embeddings_differ() {
    let pv = verdict_test_pv(false, 10, 4, true, None, 2.5, true, 0.6, true, 0.6);
    // 4 windows, only 1 matches (25% < 50%) with first mismatch at index 1.
    let cmp = test_comparison(4, 1);
    let (bucket, ev) = classify_localization_bucket(&pv, &cmp, Some(0.9));
    assert_eq!(bucket, LocalizationBucket::EmbeddingsDiffer);
    assert_eq!(cmp.first_mismatch_idx, Some(1));
    let idx = ev
        .get("first_mismatch_idx")
        .expect("first mismatch idx present");
    assert_eq!(idx.as_u64(), Some(1));
}

/// (c) vs (d) boundary: embeddings match, streaming mean < training P10 →
/// (c) embeddings_match_scores_low; mean ≥ P10 but the rolling gate never
/// accumulated → (d).
#[test]
fn localization_bucket_c_vs_d_boundary() {
    // Matched embeddings (hash_match_frac = 1.0, first_mismatch None).
    let cmp = test_comparison(3, 3);
    // Low scores: one frame scoring 0.5 (below training P10 = 0.9).
    let mut low = verdict_test_pv(false, 10, 1, true, None, 1.0, false, 0.0, true, 0.6);
    low.per_frame_scores = vec![[0.5, 1.0, 5.0]];
    let (bucket, ev) = classify_localization_bucket(&low, &cmp, Some(0.9));
    assert_eq!(bucket, LocalizationBucket::EmbeddingsMatchScoresLow);
    assert_eq!(
        ev.get("streaming_mean_score").and_then(|v| v.as_f64()),
        Some(0.5)
    );
    // Adequate scores (frame scores 0.9 >= P10 0.9) but the rolling gate never
    // reached MIN_CLASSIFIER_THRESHOLD (2.13).
    let mut adequate = verdict_test_pv(false, 10, 1, true, None, 1.0, false, 0.0, true, 0.6);
    adequate.per_frame_scores = vec![[0.9, 1.0, 5.0]];
    let (bucket, ev) = classify_localization_bucket(&adequate, &cmp, Some(0.9));
    assert_eq!(
        bucket,
        LocalizationBucket::ScoresAdequateGateNeverAccumulated
    );
    assert_eq!(
        ev.get("max_rolling_sum").and_then(|v| v.as_f64()),
        Some(1.0)
    );
}

/// (e): embeddings match, scores adequate, the rolling gate accumulated
/// (max_rolling_sum >= 2.13) but the verifier rejected → (e).  AGC
/// non-convergence is a correlated note, NOT a bucket change.
#[test]
fn localization_bucket_e_gate_passed_verifier_rejected_with_agc_note() {
    let cmp = test_comparison(3, 3);
    let pv = verdict_test_pv(false, 10, 3, true, Some(false), 2.5, true, 0.3, true, 0.6);
    let (bucket, ev) = classify_localization_bucket(&pv, &cmp, Some(0.9));
    assert_eq!(bucket, LocalizationBucket::GatePassedVerifierRejected);
    assert!(
        ev.get("agc_note").is_some(),
        "AGC non-convergence must be attached as evidence, not a bucket"
    );
}

/// Precedence: detected wins over every miss bucket.
#[test]
fn localization_bucket_detected_wins() {
    let pv = verdict_test_pv(true, 10, 3, true, Some(true), 2.5, true, 0.9, true, 0.6);
    let cmp = test_comparison(3, 0); // would otherwise be (b)
    let (bucket, _ev) = classify_localization_bucket(&pv, &cmp, Some(0.9));
    assert_eq!(bucket, LocalizationBucket::Detected);
}

/// The five buckets + detected are exactly the label space emitted in JSON —
/// every `as_str` label is unique and stable.
#[test]
fn localization_bucket_labels_are_stable() {
    let labels: Vec<&str> = [
        LocalizationBucket::Detected,
        LocalizationBucket::EmbeddingsNeverReachedClassifier,
        LocalizationBucket::EmbeddingsDiffer,
        LocalizationBucket::EmbeddingsMatchScoresLow,
        LocalizationBucket::ScoresAdequateGateNeverAccumulated,
        LocalizationBucket::GatePassedVerifierRejected,
    ]
    .iter()
    .map(|b| b.as_str())
    .collect();
    assert_eq!(labels.len(), 6);
    let mut sorted = labels.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), labels.len(), "bucket labels must be unique");
}

// ── mahbot-1022 B-sweep unit tests ─────────────────────────────────────────
// Pin the pure measurement helpers (frame distances, dedup epsilon,
// true-unique counting, ring-4 verifier windows, weight fingerprints,
// negatives extraction, verdict rule).  The model-dependent sweep driver
// itself is exercised by the full benchmark, not by these tests.

#[test]
#[expect(
    clippy::float_cmp,
    reason = "exact-zero frame distance is the pinned behavior"
)]
fn bsweep_frame_relative_l2_basics() {
    let a = [1.0_f32, 2.0, 3.0];
    // Identical → 0.
    assert_eq!(frame_relative_l2(&a, &a), 0.0);
    // Scaled (b = 2a) → relative distance ||a-b||/||b|| = 0.5 (small relative).
    let scaled = [2.0_f32, 4.0, 6.0];
    let rel = frame_relative_l2(&a, &scaled);
    assert!(
        (rel - 0.5).abs() < 1e-5,
        "scaled → ~0.5 relative, got {rel}"
    );
    // Orthogonal → > 0 (sqrt(2) with unit denominators).
    let orth = frame_relative_l2(&[1.0_f32, 0.0, 0.0], &[0.0, 1.0, 0.0]);
    assert!(orth > 0.0, "orthogonal → > 0, got {orth}");
    assert!((orth - std::f32::consts::SQRT_2).abs() < 1e-5);
}

#[test]
#[expect(
    clippy::float_cmp,
    reason = "the uniform-no-split epsilon collapse to exactly 0.0 is the pinned behavior"
)]
fn bsweep_derive_dedup_epsilon_bimodal_vs_uniform() {
    // Bimodal: duplicate cluster near 0.001, distinct cluster near 0.1.
    let mut d1 = vec![0.0008_f32, 0.0009, 0.0010, 0.0012, 0.09, 0.11, 0.12];
    let (eps, rule) = derive_dedup_epsilon(&mut d1);
    // Sorted: 0.0008, 0.0009, 0.0010, 0.0012, 0.09, 0.11, 0.12.
    // Ratios: 1.125, 1.111, 1.2, 75.0, 1.222, 1.091 → largest gap at
    // 0.0012 → 0.09 (ratio 75 ≥ 2.0) → epsilon = sqrt(0.0012 * 0.09).
    let expected = (0.0012_f32 * 0.09).sqrt();
    assert!((eps - expected).abs() < 1e-6, "gap midpoint, got {eps}");
    assert!(
        rule.contains("bimodal"),
        "rule must document the bimodal split"
    );
    // Uniform: no 2× gap → epsilon 0.0 (tolerance ≡ bit-exact).
    let mut d2 = vec![0.01_f32, 0.011, 0.012, 0.013, 0.014, 0.015];
    let (eps2, rule2) = derive_dedup_epsilon(&mut d2);
    assert_eq!(eps2, 0.0);
    assert!(
        rule2.contains("bit-exact"),
        "rule must state the bit-exact collapse"
    );
}

#[test]
#[expect(
    clippy::float_cmp,
    reason = "the isolated-zero no-split epsilon collapse to exactly 0.0 is pinned"
)]
fn bsweep_derive_dedup_epsilon_exact_zero_floor_does_not_collapse() {
    // Regression: exact-zero distances (bit-identical frames) sit at the FLOOR
    // of the duplicate cluster.  The old rule treated the 0 → first-nonzero
    // boundary as an infinite gap, which always won AND produced a degenerate
    // epsilon sqrt(0 * next) = 0 — collapsing the tolerance to bit-exact even
    // when a clear bimodal split exists ABOVE the zeros (mahbot-1022 reviewer).
    let mut d = vec![0.0_f32, 0.0, 0.0005, 0.0006, 0.09, 0.11];
    let (eps, rule) = derive_dedup_epsilon(&mut d);
    // Sorted: 0.0, 0.0, 0.0005, 0.0006, 0.09, 0.11.  Zero pairs are skipped;
    // positive ratios: 0.0006/0.0005 = 1.2, 0.09/0.0006 = 150, 0.11/0.09 =
    // 1.222 → largest gap at 0.0006 → 0.09 → epsilon = sqrt(0.0006 * 0.09).
    let expected = (0.0006_f32 * 0.09).sqrt();
    assert!(
        (eps - expected).abs() < 1e-7,
        "bimodal split above the zero floor, got {eps}"
    );
    assert!(rule.contains("bimodal"), "rule must stay bimodal");
    assert!(
        eps > 0.0,
        "epsilon must NOT collapse to 0 on exact-zero distances"
    );

    // Isolated exact zero with no near-duplicate cluster → no split → bit-exact.
    let mut d2 = vec![0.0_f32, 0.5, 0.51, 0.52];
    let (eps2, rule2) = derive_dedup_epsilon(&mut d2);
    assert_eq!(eps2, 0.0);
    assert!(rule2.contains("bit-exact"));
}

#[test]
fn bsweep_count_true_unique_duplicate_pair() {
    // 5 frames: frame 1 and 2 are bit-identical duplicates; all others distinct.
    let mel: Vec<Vec<f32>> = vec![
        vec![0.0_f32; 32],
        vec![0.5; 32],
        vec![0.5; 32], // bit-identical duplicate of frame 1
        vec![0.7; 32],
        vec![0.9; 32],
    ];
    // Window [0..5): bit-exact → frame1 dup frame2; others distinct → 1 + 3 = 4.
    let (bit, tol_bit) = count_true_unique(&mel, 0, 5, 0.0);
    assert_eq!(bit, 4);
    assert_eq!(tol_bit, bit, "epsilon == 0 → tolerance ≡ bit-exact");
    // Tolerance eps = 0.5: frame0→1 (rel 1.0 > 0.5) distinct; 1→2 dup;
    // 2→3 (rel ~0.286) dup; 3→4 (rel ~0.222) dup → 1 + 1 = 2.
    let (_b, tol) = count_true_unique(&mel, 0, 5, 0.5);
    assert_eq!(tol, 2);
}

/// A trained [`VoiceVerifier`] with zeroed weights — deterministic, input
/// independent (sigmoid(fc_bias)) — for fingerprint/ring-4 tests.
#[cfg(test)]
fn test_bsweep_verifier(threshold: f32) -> VoiceVerifier {
    use crate::audio::voice_verifier::VerifierActivation;
    VoiceVerifier {
        conv_weight: vec![0.0; 2 * 96 * 3],
        conv_bias: vec![0.0; 2],
        fc_weight: vec![0.0; 2],
        fc_bias: vec![0.0; 1],
        activation: VerifierActivation::LeakyReLU,
        threshold,
        trained: true,
        ensemble_members: Vec::new(),
    }
}

#[test]
fn bsweep_ring4_window_scores_two_windows() {
    // Ring embeddings with a SINGLE nonzero dim (dim 0 = k+1) so the
    // 288-dim L2-normalization keeps the expected sigmoid values tractable.
    let mut ring: Vec<Vec<f32>> = Vec::new();
    for k in 0..4 {
        let mut emb = vec![0.0_f32; 96];
        emb[0] = k as f32 + 1.0;
        ring.push(emb);
    }
    // Untrained verifier → predict = 1.0 for every window.
    let untrained = VoiceVerifier::untrained();
    let (max, scores, fams) =
        ring4_window_scores(&ring, &untrained, 68, &[0, 8, 16, 24]).expect("ring 4 → Some");
    assert_eq!(scores.len(), 2, "ring 4 → two stride-1 windows");
    assert_eq!(scores, vec![1.0, 1.0]);
    assert!((max - 1.0).abs() < 1e-6);
    // Real-frame families: window 0 = ring[0..3] at positions 0,8,16 → [68,60,52];
    // window 1 = ring[1..4] at positions 8,16,24 → [60,52,44].
    assert_eq!(fams[0], vec![68, 60, 52]);
    assert_eq!(fams[1], vec![60, 52, 44]);

    // Ring length != 4 → None (verifier NOT evaluated — never a 0.0 sentinel).
    let short = vec![vec![0.0_f32; 96]; 3];
    assert!(
        ring4_window_scores(&short, &untrained, 68, &[0, 8, 16]).is_none(),
        "ring < 4 → None"
    );

    // Trained verifier with a discriminating conv weight: window 0 (values
    // 1,2,3) scores higher than window 1 (values 2,3,4) — sigmoid(5/(3√14))
    // vs sigmoid(7/(3√29)); the max must be window 0's score.
    let mut v = test_bsweep_verifier(0.86);
    v.conv_weight[2] = 1.0; // channel 0, dim 0, kernel offset 2 → picks x[0][li+1]
    v.fc_weight[0] = 1.0;
    let (max, scores, _) =
        ring4_window_scores(&ring, &v, 68, &[0, 8, 16, 24]).expect("ring 4 → Some");
    assert_eq!(scores.len(), 2);
    let w0 = 1.0 / (1.0 + (-(5.0_f32 / (3.0 * 14.0_f32.sqrt()))).exp());
    let w1 = 1.0 / (1.0 + (-(7.0_f32 / (3.0 * 29.0_f32.sqrt()))).exp());
    assert!(
        (scores[0] - w0).abs() < 1e-5,
        "window 0 = {w0}, got {}",
        scores[0]
    );
    assert!(
        (scores[1] - w1).abs() < 1e-5,
        "window 1 = {w1}, got {}",
        scores[1]
    );
    assert!(scores[0] > scores[1]);
    assert!(
        (max - scores[0]).abs() < 1e-6,
        "max must be window 0's score"
    );
}

#[test]
fn bsweep_weights_fingerprints_deterministic_and_distinguishing() {
    use crate::audio::wake_word_classifier::{ArchConfig, ClassifierWeights};
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let w1 = ClassifierWeights::from_rng(&mut rng, &ArchConfig::default());
    // Deterministic.
    assert_eq!(
        weights_fingerprint_classifier(&w1),
        weights_fingerprint_classifier(&w1)
    );
    // Different weights → different hash.
    let mut w2 = w1.clone();
    w2.fc_bias[0] += 0.01;
    assert_ne!(
        weights_fingerprint_classifier(&w1),
        weights_fingerprint_classifier(&w2)
    );
    // BN running stats are part of the hash (item 6: INCLUDED).
    let mut w3 = w1.clone();
    w3.bn1_running_mean[0] += 0.1;
    assert_ne!(
        weights_fingerprint_classifier(&w1),
        weights_fingerprint_classifier(&w3)
    );

    // Verifier: threshold included.
    let v1 = test_bsweep_verifier(0.86);
    let v2 = test_bsweep_verifier(0.5);
    let h1 = weights_fingerprint_verifier(&v1);
    assert_eq!(h1, weights_fingerprint_verifier(&v1), "deterministic");
    assert_ne!(
        h1,
        weights_fingerprint_verifier(&v2),
        "threshold must be hashed"
    );
    let h = weights_fingerprint_verifier(&v1);
    assert_eq!(h.len(), 64, "sha2-256 hex digest");
    assert!(h.bytes().all(|b| b.is_ascii_hexdigit()));
}

#[test]
fn bsweep_negative_verifier_extraction_first_crossing_and_peak() {
    // Crossing variant: verdict_test_pv defaults per_frame_scores to
    // [[0.9, 2.2, 2.0]] — rolling sum 2.2 >= 2.13 → first crossing at frame 0.
    let crossing = verdict_test_pv(false, 10, 1, true, None, 2.5, true, 0.7, true, 0.6);
    // Non-crossing variant: override per_frame_scores below the gate.
    let mut no_cross = verdict_test_pv(false, 10, 1, true, None, 1.0, false, 0.2, true, 0.6);
    no_cross.per_frame_scores = vec![[0.1, 0.5, 5.0]];
    no_cross.verifier_score_trajectory = vec![0.2];
    let all: Vec<(&PerVariantResult, String)> = vec![
        (&crossing, "confusable_easy".to_string()),
        (&no_cross, "unrelated".to_string()),
    ];
    let json = negative_verifier_extraction(&all);
    assert_eq!(json["total_negatives"], 2);
    assert_eq!(json["gate_crossing_count"], 1);
    let crossing_json = &json["per_crossing"][0];
    assert_eq!(crossing_json["first_crossing_frame_idx"], 0);
    // f32 0.7 serializes as f64::from(0.7f32) — compare in f64 space.
    assert_eq!(
        crossing_json["first_crossing_verifier_value"].as_f64(),
        Some(f64::from(0.7_f32)),
    );
    assert_eq!(
        crossing_json["peak_verifier_value"].as_f64(),
        Some(f64::from(0.7_f32)),
    );
    // Per-variant peaks include both variants.
    assert_eq!(
        json["per_variant_peak_verifier"].as_array().map(Vec::len),
        Some(2)
    );
}

#[test]
fn bsweep_verdict_passes_4of5_and_unmeasurable_fallback() {
    // 5 measurable, 4 confirm → pass.
    assert_eq!(
        bsweep_verdict_passes(&[Some(true), Some(true), Some(true), Some(true), Some(false)]),
        (4, 5, true),
    );
    // 5 measurable, 3 confirm → fail.
    assert_eq!(
        bsweep_verdict_passes(&[Some(true), Some(true), Some(true), Some(false), Some(false)]),
        (3, 5, false),
    );
    // Unmeasurable fallback: 4 measurable (3 confirm + 1 reject) → 3 < 4 → fail.
    assert_eq!(
        bsweep_verdict_passes(&[Some(true), Some(true), Some(true), Some(false), None]),
        (3, 4, false),
    );
    // Unmeasurable fallback: 4 measurable, all confirm → 4-of-4 measurable → pass.
    assert_eq!(
        bsweep_verdict_passes(&[Some(true), Some(true), Some(true), Some(true), None]),
        (4, 4, true),
    );
    // All unmeasurable → fail.
    assert_eq!(bsweep_verdict_passes(&[None; 5]), (0, 0, false));
}

#[test]
fn bsweep_cross_check_scopes_pass_to_full_length_windows() {
    // F1 trajectory (mirrors the post-deferral shape): burst/pass padded
    // windows (padded_fallback, 44-68 rf) + a full-length misaligned window
    // at start 3 (true_sliding, 76 rf) + post-trim re-score windows over the
    // trimmed buffer (padded_fallback, 68/60/52 rf).
    let mut cmp = test_comparison(4, 0);
    cmp.label = "F1.json_enroll0".to_string();
    cmp.streaming.window_start = vec![0, 8, 3, 0];
    cmp.streaming.real_frames = vec![68, 60, 76, 68];
    cmp.streaming.geometry = vec![
        "padded_fallback",
        "padded_fallback",
        "true_sliding",
        "padded_fallback",
    ];
    let mut report = Mahbot1012Report::default();
    report.training_clip_comparisons.push(cmp);

    // Sweep recomputes a 79-frame shared mel: the start-3 full-length window
    // expects min(79-3, 76) = 76 → agree.  The padded windows are
    // informational and must NOT force pass=false (the structural-bug fix).
    let json = bsweep_cross_check(&report, 79);
    assert_eq!(
        json["pass"], true,
        "full-length window agrees, padded windows informational"
    );
    assert_eq!(json["n_comparable_windows"], 1);
    assert_eq!(json["n_agreeing_windows"], 1);
    let w3 = &json["windows"][2];
    assert_eq!(w3["recorded_start"], 3);
    assert_eq!(w3["expected_real_frames"], 76);
    assert_eq!(w3["agree"], true);
    let w_padded = &json["windows"][0];
    assert_eq!(
        w_padded["comparable"], false,
        "burst/pass padded window is informational"
    );
    assert!(w_padded["note"].is_string());

    // Divergent full-length window: the recomputed mel is only 74 frames →
    // start-3 expects min(74-3, 76) = 71 vs recorded 76 → delta 5 → pass=false.
    let json = bsweep_cross_check(&report, 74);
    assert_eq!(
        json["pass"], false,
        "full-length divergence beyond ±3 flags pass"
    );
    assert_eq!(json["n_comparable_windows"], 1);
    assert_eq!(json["n_agreeing_windows"], 0);
    assert_eq!(json["windows"][2]["agree"], false);

    // No F1 comparison → no comparable windows → pass=false (unverifiable).
    let empty = Mahbot1012Report::default();
    let json = bsweep_cross_check(&empty, 79);
    assert_eq!(json["pass"], false);
    assert_eq!(json["n_comparable_windows"], 0);
}

// ── Non-cached generation helpers (fallback when cache unavailable) ──────
