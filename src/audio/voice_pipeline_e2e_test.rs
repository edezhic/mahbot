//! E2E integration test / benchmark for the full voice pipeline.
//!
//! This test exercises the **enrollment-to-detection cycle with realistic
//! TTS-generated speech audio**.  It uses the TTS engine to synthesize wake
//! word variants, feeds them through the enrollment pipeline, builds the
//! prototype enrollment (L2-normalized centroid of utterance embeddings via
//! the shared Qwen3-ASR encoder), calibrates it against negative samples,
//! then runs detection on:
//!
//! * Positive cases (wake word variants)
//! * Negative — confusable near-miss phrases
//! * Negative — completely unrelated speech
//! * Negative — silence and noise
//!
//! # Running as benchmark (canonical — minimal feature set)
//!
//! The minimal invocation excludes the app binary from the build (the bin is
//! gated behind the default-on `full` feature via `required-features` on
//! `[[bin]]`, so `--no-default-features` skips its full-program link).
//! This is the documented fast path for all bench runs:
//!
//! ```sh
//! cargo bench --no-default-features --features voice-tests --bench wake_word
//! ```
//!
//! A full-feature run (builds the app binary too — much slower) is:
//!
//! ```sh
//! cargo bench --features voice-tests --bench wake_word
//! ```
//!
//! NOTE: `--no-default-features` skips the app binary for ALL target
//! selections (`cargo build`/`check`/`test`/`install`), not just bench — this
//! is inherent to the `required-features` mechanism (bench-only bin exclusion
//! is not supported by Cargo; `bench = false` does not work, rust-lang/cargo#15702).
//! Default builds are unaffected (`full` is default-on; the bin always builds).
//!
//! NOTE: `[profile.bench]` is release-like (opt-level 2, codegen-units 32,
//! no LTO, incremental off) — the bench is a production-performance proxy, so
//! release codegen is the faithful target.  Acceptance metrics are report-only
//! (measured, never gated).
//!
//! # Fixed benchmark phrase
//!
//! Every run tests the fixed bench phrase **"hey mahbot"** — never resolved
//! from the live config store, never overridable via env.  This keeps
//! benchmark numbers comparable across runs and guarantees the live personal
//! enrollment phrase can never drive benchmark runs.
//!
//! # TTS caveat
//!
//! Recognition and the synthetic false-reaction set are measured on
//! synthesized (TTS) speech, not real human speech; real audio is used for
//! the false-reaction rate only.  The report discloses this.
//!
//! First run populates the TTS audio cache (subsequent runs hit it).  The
//! encoder pipeline re-encodes raw audio through the shared Qwen3-ASR model
//! per run — there is no embedding disk cache — so the wall clock is dominated
//! by encoder forwards over the TEST surface (40-clip wake-only basis, the
//! 113 non-phrase set, and the parallel real-audio feed).  The exact wall
//! clock is auditable from the top-level `wall_clock_secs` key (whole run,
//! report-assembly window included).
//!
//! # Requirements
//!
//! * TTS models must be downloaded and cached (run the app once).
//! * The shared Qwen3-ASR model must be present in `~/.mahbot/models/`
//!   (the same model the local transcriber loads — wake word detection
//!   reuses it; the old OpenWakeWord ONNX models are no longer used).

// Clippy allowances — use super::* is intentional (mirrors the voice module's API surface).
// Cast warnings: this is a benchmark file with known-safe numeric conversions.
#![allow(
    clippy::wildcard_imports,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use super::*; // voice module items (handle_wake_word_detection, PipelineCtx, etc.)
use crate::audio::tts;
use crate::util::hex_string;
use earshot::Detector;
use rand::{RngExt, SeedableRng};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::path::Path;
use std::time::Instant;

// ── Constants ──────────────────────────────────────────────────────────────

/// Number of samples in the warm-up audio prepended before each test
/// utterance.
///
/// ~1.28 s of background audio at 16 kHz fed through the pipeline before the
/// test utterance so the warm pass exercises the ring-primed, segment-warmed
/// path (production background silence/noise before anyone speaks).
const WARMUP_PREPEND_SAMPLES: usize = 20480; // 1.28s × 16 kHz

/// TTS phrase for warm-up audio.
///
/// A short non-wake-word phrase synthesised via the already-loaded TTS engine.
/// Speech-like harmonics guarantee the Earshot neural VAD triggers, producing
/// embeddings for the pre-utterance ring.
///
/// Must NOT contain the wake word ([`BENCH_WAKE_PHRASE`]) or phonetically
/// similar phrases that could trigger detection.
const WARMUP_TTS_PHRASE: &str = "testing one two three";

/// Cached TTS warm-up audio, populated on first successful synthesis.
/// Unlike the original `WARMUP_NOISE_CACHE`, this caches ONLY the TTS
/// result — if TTS is unavailable on the first call, a fresh pink-noise
/// fallback is returned (no caching), so TTS is re-evaluated on subsequent
/// calls (avoids cache poisoning).
static WARMUP_TTS_CACHE: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();

/// The fixed wake phrase every benchmark run synthesizes and tests.
///
/// Pinned so benchmark numbers stay comparable across runs and the live
/// personal enrollment phrase can never drive benchmark runs (the bench
/// never reads the deployed config-store phrase).
const BENCH_WAKE_PHRASE: &str = "hey mahbot";

/// Number of enrollment variants to generate.
///
/// Production generates ~110-120 positive training sequences from 10
/// utterances × the full 12-cell recipe (minus SpeedUp skip for
/// short utterances).  We target ~10 TTS enrollment variants to match this.
/// Since the re-scope all 10 clips come from ONE TTS voice (the enrolled
/// speaker) with guided-prompt DSP conditioning.
const NUM_ENROLLMENT_VARIANTS: usize = 10;

/// Owner-negative (non-wake-word) phrases for negative calibration
/// (re-scoped).
///
/// These are generated via TTS from the ENROLLED voice only (the owner's own
/// non-wake-word speech) and encoded as negative samples to help the
/// prototype reject general speech from the enrolled user.  The phrase list is
/// DISJOINT from the confusable/unrelated pools so no phrase is double-labeled
/// across training pools (the old list overlapped the unrelated pool:
/// 'turn on the lights' / 'what time is it' exact matches, 'good morning' vs
/// 'good morning everyone').  Documented limitation: TTS speech cannot match
/// the distribution of real human Phase 3 speech collected during production
/// enrollment.
const OWNER_NEGATIVE_PHRASES: &[&str] = &[
    "please stop",
    "i am going out",
    "where are my keys",
    "the meeting starts soon",
    "let me check the weather",
    // Disjoint non-wake phrases, collision-free seeds.
    "remind me to call the dentist",
    "what should we have for dinner",
    "the wifi keeps disconnecting",
    "i need to buy new shoes",
    "turn down the music please",
];

/// Single-voice allocation for the bench.
///
/// The standard 10-style set is F1-F5, M1-M5 in list order.  The allocation
/// (by index into `available_styles`):
/// - enrolled: index 0 (F1) — the ONLY voice in the enrollment clips and
///   owner negatives.
/// - negative pool: indices 0..6 (F1..M1) — the confusable/unrelated prewarm
///   styles.  Production rotates ALL voices; the bench pins F1 for the
///   enrollment so the negative pool never trains in.
struct VoiceAllocation {
    enrolled: String,
    negative_pool: Vec<String>,
}

/// Derive the voice allocation from the available TTS styles (defensive for
/// machines with fewer than 10 styles: the negative pool degrades to all
/// styles when fewer than 6 are present).
fn allocate_voices(available_styles: &[String]) -> VoiceAllocation {
    let enrolled = available_styles
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_TTS_STYLE.to_string());
    let take = |range: std::ops::Range<usize>| {
        available_styles
            .get(range)
            .map(<[String]>::to_vec)
            .unwrap_or_default()
    };
    VoiceAllocation {
        enrolled: enrolled.clone(),
        negative_pool: {
            // indices 0..6 (F1..M1) when present; otherwise all styles (a
            // fewer-than-6 machine's negative pool degrades to all styles).
            let pool = take(0..6);
            if pool.is_empty() {
                available_styles.to_vec()
            } else {
                pool
            }
        },
    }
}

/// Guided-enrollment prompt groups (production [`ENROLLMENT_PROMPTS`]:
/// 3 normal, 3 "further from the mic", 2 "different angle", 2 "morning voice").
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GuidedPromptGroup {
    Normal,
    Distance,
    Angle,
    Morning,
}

impl GuidedPromptGroup {
    /// Map an enrollment clip index (0..10) to its guided prompt group.
    const fn for_clip_index(i: usize) -> Self {
        match i {
            0..=2 => Self::Normal,
            3..=5 => Self::Distance,
            6..=7 => Self::Angle,
            _ => Self::Morning,
        }
    }

    /// Detection SNR against the realistic noise floor (dB).  The relationship
    /// SNR(level_reduction) is chosen so every group's noise lands at the SAME
    /// absolute RMS as the normal 20 dB floor (distance −6 dB + 14 dB SNR,
    /// angle/morning −3 dB + 17 dB SNR) — the encoder pipeline consumes raw
    /// audio, so realism must come from the SNR / spectral relationships.
    const fn noise_snr_db(self) -> f32 {
        match self {
            Self::Normal => 20.0,
            Self::Distance => 14.0,
            Self::Angle | Self::Morning => 17.0,
        }
    }
}

/// Confusable near-miss phrases for negative detection testing.
///
/// Bench-local copy of the canonical mahbot-family confusable list, kept for
/// negative-audio generation.
const CONFUSABLE_PHRASES: &[&str] = &[
    // ── Direct phonetic substitutions (wake-word-like) ──
    "hey madbot",
    "hey map bot",
    "day mahbot",
    "hey nab it",
    "hey man",
    "hey mabot",
    "hey mahbott",
    "hey mat",
    "hey max",
    "pay mabot",
    // ── Rhythmic/melodic confusables ──
    "hay map pot",
    "huh mahbot",
    "eh mad bot",
    "hey maybott",
    "they mad bot",
    "haymaker",
    // ── Embedded wake-word sounds ──
    "hey maybe not",
    "play mah jong",
    "hey matter of fact",
    "a day with mahbot",
    // ── Short phonetic near-misses ──
    "madbot",
    "mat bot",
    "bad bot",
    "mad lot",
    "mad pot",
    "med bot",
    "my bot",
    "may bot",
];

/// Seeds per confusable phrase and the base seed (bench-local).
const CONFUSABLE_SEEDS_PER_PHRASE: usize = 5;
const CONFUSABLE_SEED_BASE: u64 = 1000;

/// Unrelated speech phrases for negative detection testing.
///
/// Bench-local copy of the canonical unrelated-speech list.
const UNRELATED_PHRASES: &[&str] = &[
    // ── Short commands (2-4 words) ──────────────────────────────────
    "the weather today is sunny",
    "what time is it",
    "one two three four five",
    "hello world",
    "good morning everyone",
    "turn on the lights",
    "play some music",
    "set a timer",
    // ── Medium phrases (5-8 words) ───────────────────────────────────
    "i need to buy groceries today",
    "can you remind me of my appointment",
    "please send a message to john",
    "what is the capital of france",
    "tell me a joke about programming",
    "how do I get to the airport",
    "the quick brown fox jumps over the lazy dog",
    // ── Long utterances (10+ words) ──────────────────────────────────
    "according to all known laws of aviation there is no way a bee should be able to fly",
    "the principle of superposition states that a quantum system exists in all its possible states simultaneously",
    // ── Non-English phrases (phonetically distinct from English wake word) ──
    "bonjour comment allez vous aujourd hui",
    "buenos días cómo estás",
    "guten morgen wie geht es dir",
];

/// Seeds per unrelated phrase and the base seed — bench-local.
const UNRELATED_SEEDS_PER_PHRASE: usize = 3;
const UNRELATED_SEED_BASE: u64 = 2000;

/// Silence durations for the negative silence matrix: three
/// durations in samples at 16 kHz (0.5 s / 1.0 s / 2.0 s).
const SILENCE_DURATIONS: &[(&str, usize)] = &[
    ("silence_0_5s", 8_000),
    ("silence_1_0s", 16_000),
    ("silence_2_0s", 32_000),
];

/// Noise audio length in samples (1 second at 16 kHz).
const NOISE_LEN: usize = 16_000;

/// Noise profiles for negative detection testing.
///
/// Each noise profile is a (label, generator_fn) pair.  The generator
/// produces PCM f32 samples at 16 kHz.
type NoiseGenerator = fn() -> Vec<f32>;

/// Training noise profiles.  This list feeds negative calibration via
/// [`generate_ambient_noise_sequences`] — the detection-only additions
/// ([`NOISE_PROFILES_DETECTION_ONLY`]) must never be appended here.
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

/// Detection-only noise profiles: fresh generator seeds 52-55.  Chained after
/// [`NOISE_PROFILES`] at the detection call sites — never appended to the
/// training list.
const NOISE_PROFILES_DETECTION_ONLY: &[(&str, NoiseGenerator)] = &[
    ("impulse burst noise", generate_impulse_burst_noise),
    ("oscillating tone noise", generate_oscillating_tone_noise),
    ("crackle static noise", generate_crackle_static_noise),
    ("pulsed broadband noise", generate_pulsed_broadband_noise),
];

/// All noise profiles for the detection call sites: the training list plus
/// the detection-only additions (training never sees the latter).  Single
/// owner of the "detection enumerations include the detection-only profiles"
/// invariant.
fn all_noise_profiles() -> impl Iterator<Item = &'static (&'static str, NoiseGenerator)> {
    NOISE_PROFILES.iter().chain(NOISE_PROFILES_DETECTION_ONLY)
}

/// TTS target sample rate (voice pipeline rate).
const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Default TTS voice style when no styles are available from disk.
/// This matches the naming convention used by the TTS model download.
const DEFAULT_TTS_STYLE: &str = "M1.json";

/// Cache directory relative to storage root.
const TEST_CACHE_DIR: &str = "test_cache/voice_e2e";

// ── TTS audio cache ─────────────────────────────────────

// Compute a deterministic model version hash from TTS compile-time SHA-256
// constants.
//
// Hashes the concatenation of all TTS model SHA-256 digests and the voice
// style directory hash.  These constants are updated in `tts.rs` whenever
// model files change, so the cache key automatically tracks model version
// without disk I/O.
//
// Returns a hex string (always succeeds since the hashes are compile-time).
//
// Delegates to [`tts_model_version_hash`], the canonical implementation.

/// Get the cache directory path.
fn cache_dir() -> std::path::PathBuf {
    let root = crate::config::CONFIG
        .try_storage_root()
        .expect("CONFIG storage root must be set");
    root.join(TEST_CACHE_DIR)
}

// ── Voice PCM disk cache helpers ─────────────────────────────
//
// Bench scaffolding that synthesises TTS PCM and caches it on disk.
// Nothing in the production detection path synthesises TTS PCM, so these
// helpers live here (voice-tests-only) rather than in `voice.rs`.

/// Deterministic cache key for one TTS-synthesised PCM utterance.
///
/// Covers: text, TTS style, seed, sample rate, and the TTS model version
/// hash.  A change to any TTS model file produces a different key and forces
/// re-synthesis on first run.
fn pcm_cache_key(text: &str, style: &str, seed: u64, sample_rate: u32, model_hash: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hasher.update([0u8]);
    hasher.update(style.as_bytes());
    hasher.update([0u8]);
    hasher.update(seed.to_le_bytes());
    hasher.update([0u8]);
    hasher.update(sample_rate.to_le_bytes());
    hasher.update([0u8]);
    hasher.update(model_hash.as_bytes());
    hex_string(&hasher.finalize())
}

/// Hash of all TTS model SHA256 constants for cache invalidation.
///
/// Any change to any TTS model file produces a different hash, which
/// changes the PCM cache key and triggers re-synthesis on first run.
fn tts_model_version_hash() -> String {
    let mut hasher = Sha256::new();
    hasher.update(crate::audio::tts::DP_MODEL_SHA256.as_bytes());
    hasher.update(crate::audio::tts::TEXT_ENC_MODEL_SHA256.as_bytes());
    hasher.update(crate::audio::tts::VECTOR_EST_MODEL_SHA256.as_bytes());
    hasher.update(crate::audio::tts::VOCODER_MODEL_SHA256.as_bytes());
    hasher.update(crate::audio::tts::TTS_JSON_SHA256.as_bytes());
    hasher.update(crate::audio::tts::UNICODE_INDEXER_SHA256.as_bytes());
    hasher.update(crate::audio::tts::VOICE_STYLE_SHA256.as_bytes());
    hex_string(&hasher.finalize())
}

/// Write PCM f32 samples to the disk cache atomically (tmp + rename).
fn write_pcm_cache(path: &Path, samples: &[f32]) {
    let tmp_path = path.with_extension("tmp");
    let mut data: Vec<u8> = Vec::with_capacity(samples.len() * 4);
    for &s in samples {
        data.extend_from_slice(&s.to_le_bytes());
    }
    if let Err(e) = std::fs::write(&tmp_path, &data) {
        warn!(
            "PCM cache: failed to write tmp file {}: {e}",
            tmp_path.display()
        );
        return;
    }
    if let Err(e) = std::fs::rename(&tmp_path, path) {
        warn!(
            "PCM cache: failed to rename {} -> {}: {e}",
            tmp_path.display(),
            path.display(),
        );
    }
}

/// Read PCM f32 samples from the disk cache.
///
/// Returns `None` if the cache file does not exist, has non-aligned size
/// (not a multiple of 4 bytes), or fails to read.
fn read_pcm_cache(path: &Path) -> Option<Vec<f32>> {
    let data = std::fs::read(path).ok()?;
    if data.is_empty() {
        warn!("PCM cache: file {} is empty — deleting", path.display(),);
        let _ = std::fs::remove_file(path);
        return None;
    }
    if data.len() % 4 != 0 {
        warn!(
            "PCM cache: file {} has non-aligned size {} — deleting",
            path.display(),
            data.len(),
        );
        let _ = std::fs::remove_file(path);
        return None;
    }
    let samples: Vec<f32> = data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Some(samples)
}

/// Synthesise a phrase via TTS, caching PCM audio to disk.
///
/// Checks the voice PCM disk cache first (keyed by text + style + seed +
/// sample rate + TTS model hash).  On cache hit, returns cached PCM
/// directly without calling TTS.  On cache miss, calls
/// [`crate::audio::tts::synthesize`], writes the result to the cache, and returns
/// the PCM.
///
/// Returns `None` if TTS synthesis fails or the cache directory cannot
/// be resolved.
fn synthesize_with_pcm_cache(
    text: &str,
    style: &str,
    seed: u64,
    sample_rate: u32,
    model_hash: &str,
    cache_dir: &Path,
) -> Option<Vec<f32>> {
    let key = pcm_cache_key(text, style, seed, sample_rate, model_hash);
    let cache_path = cache_dir.join(&key);

    // Fast path: cache hit.
    if let Some(pcm) = read_pcm_cache(&cache_path) {
        debug!("PCM cache HIT for key {key} ({text}, style={style}, seed={seed})");
        return Some(pcm);
    }

    debug!("PCM cache MISS for key {key} ({text}, style={style}, seed={seed}) — synthesising");

    // Cache miss — synthesise via TTS
    let Ok(pcm) = crate::audio::tts::synthesize(text, style, seed, sample_rate) else {
        return None;
    };

    // Write to disk cache atomically
    write_pcm_cache(&cache_path, &pcm);
    Some(pcm)
}

/// Synthesize wake word variant audio with TTS caching support.
///
/// Delegates to the local [`synthesize_with_pcm_cache`] helper.
fn synthesize_wake_word_variant_cached(
    text: &str,
    style: &str,
    seed: u64,
    sample_rate: u32,
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Option<Vec<f32>> {
    synthesize_with_pcm_cache(text, style, seed, sample_rate, model_hash, cache_dir)
}

// ── Audio generation helpers (with cache) ─────────────────────────────────

// ── Guided-prompt DSP conditioning ──────────────────────────
// All primitives are deterministic (pure arithmetic or seeded RNG): the
// pinned-seed weights contract and cross-run comparability hold.

/// Deterministic first-order IIR lowpass (one-pole):
/// `y[n] = y[n-1] + α(x[n] − y[n-1])`, `α = 1 − exp(−2π fc / fs)`.
fn one_pole_lowpass(pcm: &[f32], cutoff_hz: f32, sample_rate: u32) -> Vec<f32> {
    if cutoff_hz <= 0.0 || cutoff_hz >= sample_rate as f32 * 0.5 {
        return pcm.to_vec();
    }
    let alpha = 1.0 - (-2.0 * core::f32::consts::PI * cutoff_hz / sample_rate as f32).exp();
    let mut y = 0.0f32;
    pcm.iter()
        .map(|&x| {
            y += alpha * (x - y);
            y
        })
        .collect()
}

/// Deterministic first-order high-shelf CUT (RBJ cookbook, S=1): attenuates
/// frequencies above `fc_hz` by `gain_db` — the spectral tilt for the
/// "different angle" guided-enrollment condition.
fn high_shelf_cut(pcm: &[f32], gain_db: f32, fc_hz: f32, sample_rate: u32) -> Vec<f32> {
    if gain_db.abs() < 1e-6 || fc_hz <= 0.0 {
        return pcm.to_vec();
    }
    let a = 10.0_f32.powf(gain_db / 40.0);
    let w0 = 2.0 * core::f32::consts::PI * fc_hz / sample_rate as f32;
    // RBJ high shelf with slope S = 1: alpha = sin(w0)/2 · sqrt(2).
    let alpha = w0.sin() / 2.0 * core::f32::consts::SQRT_2;
    let cos_w0 = w0.cos();
    let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
    let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
    let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
    let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
    let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha;
    let inv_a0 = 1.0 / a0;
    let (b0, b1, b2, a1, a2) = (
        b0 * inv_a0,
        b1 * inv_a0,
        b2 * inv_a0,
        a1 * inv_a0,
        a2 * inv_a0,
    );
    let mut x1 = 0.0f32;
    let mut x2 = 0.0f32;
    let mut y1 = 0.0f32;
    let mut y2 = 0.0f32;
    pcm.iter()
        .map(|&x| {
            let y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = x;
            y2 = y1;
            y1 = y;
            y
        })
        .collect()
}

/// Apply the guided-prompt DSP conditioning to one raw enrollment clip
/// Variation comes from DSP on the cached synthesis PCM — no
/// extra TTS synthesis cost.  Each group's level reduction is calibrated so
/// the added noise lands at the SAME absolute RMS as normal's 20 dB floor
/// (the encoder pipeline consumes raw audio, so realism comes from SNR vs the fixed
/// floor, not absolute level); the calibration is exact for the pure
/// level-reduction paths and approximate where the rolloff/resample shift the
/// signal RMS the SNR references:
/// - normal: realistic noise floor (20 dB SNR);
/// - distance: level reduction (−6 dB) + high-frequency rolloff (3.2 kHz
///   lowpass) + noise floor at 14 dB SNR;
/// - angle: spectral tilt (high-shelf cut −4 dB @ 3 kHz) + level reduction
///   (−3 dB) + noise floor at 17 dB SNR;
/// - morning: slower/lower-pitch resample (0.92×) + level reduction (−3 dB) +
///   lowpass (2.2 kHz) + noise floor at 17 dB SNR.
///
/// The noise floor is deterministic seeded pink noise (seed 4000 + clip index).
fn condition_enrollment_clip(pcm: &[f32], clip_index: usize) -> Vec<f32> {
    let group = GuidedPromptGroup::for_clip_index(clip_index);
    let conditioned = match group {
        GuidedPromptGroup::Normal => pcm.to_vec(),
        GuidedPromptGroup::Distance => {
            let attenuated = crate::util::apply_gain(pcm, -6.0);
            one_pole_lowpass(&attenuated, 3200.0, TARGET_SAMPLE_RATE)
        }
        GuidedPromptGroup::Angle => {
            let tilted = high_shelf_cut(pcm, -4.0, 3000.0, TARGET_SAMPLE_RATE);
            crate::util::apply_gain(&tilted, -3.0)
        }
        GuidedPromptGroup::Morning => {
            let slower = crate::util::speed_perturbation(pcm, TARGET_SAMPLE_RATE, 0.92);
            let reduced = crate::util::apply_gain(&slower, -3.0);
            one_pole_lowpass(&reduced, 2200.0, TARGET_SAMPLE_RATE)
        }
    };
    crate::util::add_noise(&conditioned, group.noise_snr_db(), 4000 + clip_index as u64)
}

/// Generate the 10 single-voice enrollment clips.
///
/// The enrolled voice (F1 in the standard set — `voice_allocation.enrolled`)
/// is synthesized once per clip at seeds 100-109; the guided-prompt DSP
/// conditioning (3 normal / 3 distance / 2 angle / 2 morning) derives ALL
/// acoustic variation from the cached PCM.  The former multi-voice rotation
/// (`style = available_styles[i % num_styles]`) is gone — it was
/// the single largest representativeness gap (the "enrolled speaker" was 6
/// different voices and the model failed production's own enrollment gate).
fn generate_enrollment_variants_cached(
    enrolled_style: &str,
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Vec<(Vec<f32>, String)> {
    let mut variants = Vec::with_capacity(NUM_ENROLLMENT_VARIANTS);
    for i in 0..NUM_ENROLLMENT_VARIANTS {
        let seed = 100 + i as u64;
        if let Some(pcm) = synthesize_wake_word_variant_cached(
            BENCH_WAKE_PHRASE,
            enrolled_style,
            seed,
            TARGET_SAMPLE_RATE,
            model_hash,
            cache_dir,
        ) {
            variants.push((
                condition_enrollment_clip(&pcm, i),
                format!("{enrolled_style}_enroll{i}"),
            ));
        }
    }
    variants
}

/// Wake-only held-out recall clip count (seeds 3000..3000+N).  Enlarged to
/// 40: the original 16 clips (3000-3015) plus 24 new clips in the freed
/// collision-free band (3016-3039).  The basis IS this enlarged set — there
/// is no separate 16-clip sub-pool.
const HELD_OUT_WAKE_ONLY_CLIPS: usize = 40;

/// Generate the held-out recall set: unseen renderings of the ENROLLED voice
/// at collision-free seeds (3000+ — avoids 100-109 enrollment, 800+ detection
/// variants, 947 warmup, 9000+ owner training, 2000+ unrelated training).
/// Generated strictly after training and never added to any training pool.
/// Wake phrase alone only — embedded-in-sentence detection is not a product
/// requirement and is not measured.
fn generate_held_out_recall_clips_cached(
    enrolled_style: &str,
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Vec<(Vec<f32>, String)> {
    let mut clips = Vec::new();
    for i in 0..HELD_OUT_WAKE_ONLY_CLIPS {
        let seed = 3000 + i as u64;
        if let Some(pcm) = synthesize_wake_word_variant_cached(
            BENCH_WAKE_PHRASE,
            enrolled_style,
            seed,
            TARGET_SAMPLE_RATE,
            model_hash,
            cache_dir,
        ) {
            clips.push((pcm, format!("{enrolled_style}_heldout_wake_s{seed}")));
        }
    }
    clips
}

/// Seed configuration for TTS phrase variant generation.
///
/// Encapsulates the three related seed parameters that control deterministic
/// TTS synthesis of phrase variants with different seeds and style rotations.
/// Bundled into a struct for call-site clarity.
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
        // (per-phrase seed formula): rotate styles across
        // (phrase × seed_variant) combos.
        let style_idx = (i * seed.num_variants + seed.seed_variant) % num_styles;
        let style = if available_styles.is_empty() {
            DEFAULT_TTS_STYLE
        } else {
            &available_styles[style_idx]
        };
        // Production seed formula:
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
        // Shared EPSILON-clamp pair sampler — preserves the
        // bench's exact draw sequence (2 draws per 2 samples, cos+sin).
        let (z1, z2) = crate::util::sample_gaussian_pair_clamped(&mut rng);
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

// Detection-only noise profiles: fresh generator seeds 52-55.  These live in
// [`NOISE_PROFILES_DETECTION_ONLY`] only — appending them to
// [`NOISE_PROFILES`] would silently extend the measured negative corpus.

/// Impulse burst noise: sparse broadband clicks.
fn generate_impulse_burst_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(52);
    (0..NOISE_LEN)
        .map(|_| {
            if rng.random::<f32>() < 0.02 {
                (rng.random::<f32>() * 2.0 - 1.0).clamp(-1.0, 1.0)
            } else {
                0.0
            }
        })
        .collect()
}

/// Oscillating tone noise: slow frequency sweep mixed with white noise.
fn generate_oscillating_tone_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(53);
    (0..NOISE_LEN)
        .map(|i| {
            let t = i as f32 / TARGET_SAMPLE_RATE as f32;
            let freq = 200.0 + 300.0 * (2.0 * core::f32::consts::PI * 0.5 * t).sin();
            let tone = (2.0 * core::f32::consts::PI * freq * t).sin() * 0.3;
            let noise: f32 = rng.random::<f32>() * 2.0 - 1.0;
            (tone + noise * 0.5).clamp(-1.0, 1.0)
        })
        .collect()
}

/// Crackle static noise: sparse high-amplitude transients over a quiet floor.
fn generate_crackle_static_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(54);
    let mut prev: f32 = 0.0;
    (0..NOISE_LEN)
        .map(|_| {
            let white = rng.random::<f32>() * 2.0 - 1.0;
            let crackle = if rng.random::<f32>() < 0.005 {
                white * 3.0
            } else {
                white * 0.1
            };
            let hi = 0.9 * (crackle - prev) + 0.8 * prev;
            prev = crackle;
            hi.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Pulsed broadband noise: white noise gated by a slow square envelope.
fn generate_pulsed_broadband_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(55);
    (0..NOISE_LEN)
        .map(|i| {
            let t = i as f32 / TARGET_SAMPLE_RATE as f32;
            let gate = if (t * 2.0).sin() > 0.0 { 1.0 } else { 0.0 };
            let noise: f32 = rng.random::<f32>() * 2.0 - 1.0;
            (noise * gate).clamp(-1.0, 1.0)
        })
        .collect()
}

/// Generate warm-up audio that reliably triggers the Earshot neural VAD.
///
/// Attempts TTS synthesis first (guaranteed speech-like harmonics), falling
/// back to pink noise + 200 Hz tone if TTS is unavailable.
///
/// Returns a [`Cow`] that either borrows from the TTS cache (once populated)
/// or owns freshly-generated pink noise (not cached, so TTS is re-evaluated
/// on the next call if it becomes available).
///
/// ## Determinism
/// TTS synthesis uses a fixed seed (947), producing the same
/// PCM output on every call (barring TTS model changes).  The result is cached
/// in [`WARMUP_TTS_CACHE`] so subsequent calls are zero-cost.
///
/// ## Fallback rationale
/// The pink noise + 200 Hz tone was the original warm-up signal,
/// but its lack of speech-like harmonic structure causes the Earshot neural VAD
/// to reject it as non-speech, producing 0 embeddings and an unprimed ring.
fn generate_warmup_noise() -> Cow<'static, [f32]> {
    // TTS cache hit — fast path, no synthesis needed.
    if let Some(cached) = WARMUP_TTS_CACHE.get() {
        return Cow::Borrowed(cached);
    }

    // First call (or cache not yet populated): try TTS synthesis.
    // Speech-like harmonics guarantee Earshot VAD triggers, producing
    // embeddings that prime the ring.
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
         This may not trigger Earshot VAD, producing 0 warm-up embeddings. \
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
        947, // fixed seed for determinism
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

/// Generate the original pink noise + 200 Hz tone warm-up signal.
///
/// Retained as a fallback when TTS models are not cached.  Properties:
/// 1. **VAD-inactive**: low probability of triggering the Earshot neural VAD
///    (no speech harmonics), which is the whole reason we now prefer TTS.
/// 2. **No false detection**: soft scores are near-zero because pink
///    noise + 200 Hz tone does not resemble the "hey mahbot" embedding.
/// 3. **Deterministic**: RNG seed 922 (= the original warm-up signal seed).
/// 4. **Aperiodic noise floor**: pink noise avoids periodic artefacts that
///    could bias the VAD steady state.
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

/// Process a single frame of audio through the pipeline
/// ([`super::handle_wake_word_detection`]), matching the per-frame processing
/// in [`run_streaming_detection`].
///
/// Partial chunks are passed through **without zero-padding**:
/// production feeds raw mic chunks as-is, and `handle_wake_word_detection`
/// accumulates partial tails in `ctx.audio_buffer` until the next real
/// chunk completes a [`FRAME_LENGTH`](super::FRAME_LENGTH) frame — the tail is
/// carried forward.  Zero-padding instead would shift the VAD/segment cadence
/// at utterance boundaries from production.
///
/// Empty slices are a no-op (the detection frame loop needs a complete frame) —
/// silence must be fed as actual [`FRAME_LENGTH`](super::FRAME_LENGTH)
/// digital-silence chunks, which is what production's mic delivers.
///
/// Extracted as a shared helper to eliminate the structural near-duplicate
/// between `feed_audio` and `run_streaming_detection`.
fn process_frame(samples: &[f32], ctx: &mut super::PipelineCtx) {
    super::handle_wake_word_detection(samples, ctx);
}

/// Feed audio through the production pipeline ([`super::handle_wake_word_detection`])
/// in [`FRAME_LENGTH`](super::FRAME_LENGTH) chunks, then send silence frames to flush any
/// remaining detection state — matching the processing pattern in [`run_streaming_detection`].
///
/// This is used to pre-warm the pipeline before the actual test utterance so
/// the ring and adaptive state are warm when the utterance starts.
fn feed_audio(samples: &[f32], ctx: &mut super::PipelineCtx) {
    for chunk in samples.chunks(super::FRAME_LENGTH) {
        process_frame(chunk, ctx);
    }
    // Feed digital-silence frames to flush any remaining detection state (matches
    // the post-audio flush in run_streaming_detection).  Real zeros in
    // FRAME_LENGTH chunks, NOT empty Vecs: production's mic keeps delivering
    // 512-sample chunks of silence after speech — an empty chunk would never
    // produce a complete frame for the detection loop.
    for _ in 0..3 {
        process_frame(&vec![0.0; super::FRAME_LENGTH], ctx);
    }
}

/// Feed the warm-up audio through [`feed_audio`] so the warm pass starts the
/// test utterance with a ring-primed, segment-warmed path.
///
/// ## Diagnostics
///
/// If the warm-up audio triggered a false detection, a `warn!()` is emitted and
/// the detection state is restored to prevent cooldown from corrupting subsequent
/// benchmark measurements.
///
/// ## Residue clearing
///
/// After feeding warm-up audio, this function clears warm-up residues that would
/// otherwise contaminate the subsequent test utterance's detection pipeline:
///
/// - **`score_window`** — cleared so warm-up soft scores cannot carry into
///   the first test embedding's rolling sum.
/// - **`segment_silence_hops`** — reset to 0 so the test utterance starts a fresh
///   segment (the warm-up flush's VAD-negative hops must not shorten the test
///   utterance's trailing-silence window).
/// - **`last_score_sample_count`** — reset so the first test-utterance scoring
///   step waits for [`SCORE_STRIDE_SAMPLES`](super::SCORE_STRIDE_SAMPLES) of
///   test audio (the warm-up feed's stride counter must not shorten it).
///
/// **Preserved**: `audio_buffer` (the trailing raw ring — the encoder window
/// spans the most recent second of audio, so warm-up context is exactly what
/// production's ring carries), `adaptive_threshold` (preserves warm-up-adapted
/// state — see the cold-start pass, which instead uses a fresh
/// [`AdaptiveThresholdState::new`]).
fn consume_warmup(ctx: &mut super::PipelineCtx) {
    let before_detection = ctx.last_wake_word_detection;
    let noise = generate_warmup_noise();

    // ── Feed warm-up audio (single pass) ───────────────────────────────────
    feed_audio(&noise, ctx);

    // Guard: warm-up audio should not trigger a false detection.  If it does,
    // restore the pre-warm-up state so cooldown doesn't suppress all subsequent
    // benchmark measurements.  We must also
    // reset `is_recording` because detection sets it to `true`, which would
    // suppress scoring for the entire benchmark variant.
    if ctx.last_wake_word_detection != before_detection {
        warn!(
            "Warm-up triggered a false detection — restoring detection \
             state to prevent cooldown corruption.",
        );
        ctx.last_wake_word_detection = before_detection;
        ctx.is_recording = false;
    }

    // ── Clear warm-up residues before test utterance processing ──
    // Warm-up audio produces soft scores and VAD hops.  These residues
    // must be cleared so the test utterance starts with a clean detection slate:
    // warm-up scores would otherwise mix into the test utterance's rolling sum,
    // and stale segment-silence hops would shorten the test window.  The raw
    // audio ring and VAD cursor are also reset so the first test-utterance
    // encoder window is not diluted by warm-up noise.
    ctx.score_window.clear();
    ctx.segment_silence_hops = 0;
    ctx.last_score_sample_count = 0;
    ctx.audio_buffer.clear();
    ctx.speech_window.clear();
    ctx.vad_cursor = 0;

    // ── Reset instrumentation ─────────────────────────────
    // Warm-up audio passes through the full scoring pipeline and records into
    // ctx.instrumentation: peak_score and the VAD-active frame count.  Without
    // a reset, warm-up-only scores contaminate the test utterance's
    // per-variant metrics (silence/noise negatives would report "detection
    // triggers" from warm-up speech).
    ctx.instrumentation = super::DetectionInstrumentation::new();
}

// ── Prerequisite check ─────────────────────────────────────────────────────

/// Ensure the shared Qwen3-ASR model is loaded (the wake-word pipeline's
/// encoder — no separate OpenWakeWord ONNX artifacts exist anymore).
/// Returns an error if loading fails, with a helpful message.
fn ensure_voice_models_loaded() -> Result<(), String> {
    if super::models_ready() {
        return Ok(());
    }

    // Await the shared transcriber load.  The benchmark runs inside a tokio
    // spawn_blocking task, so a runtime handle is usually present; fall back to
    // a fresh current-thread runtime otherwise (same pattern as
    // `faph_download_file`).
    let loaded = if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.block_on(crate::audio::local_transcriber::try_init_from_cache())
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("cannot build transcriber-init runtime: {e}"))?;
        rt.block_on(crate::audio::local_transcriber::try_init_from_cache())
    };

    if loaded {
        info!("Shared Qwen3-ASR model loaded for wake-word benchmark");
        return Ok(());
    }

    // A concurrent boot-path init may own the load — poll briefly before
    // declaring failure (the load is ~4 s with a warm page cache).
    if !crate::audio::local_transcriber::is_failed() {
        for _ in 0..75 {
            // 15 s of 200 ms polls
            if crate::audio::local_transcriber::is_loaded() {
                info!("Shared Qwen3-ASR model loaded for wake-word benchmark (concurrent init)");
                return Ok(());
            }
            if crate::audio::local_transcriber::is_failed() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    }

    Err(
        "Failed to load the shared Qwen3-ASR model (missing/corrupt cache or \
         download failure). Run the application first to download models."
            .to_string(),
    )
}

/// VAD-gate raw PCM through a dedicated earshot detector (per 256-sample hop,
/// never the global `VAD_DETECTOR`), returning only the VAD-positive hops
/// concatenated — the "VAD-gated speech" the negative generators encode.
///
/// The detector's internal state carries across hops exactly like the
/// streaming path's per-segment detector, so speech onsets/offsets are judged
/// on the continuous stream.
fn vad_gate_speech(pcm: &[f32], detector: &mut Detector) -> Vec<f32> {
    let mut speech: Vec<f32> = Vec::new();
    for chunk in pcm.chunks(256) {
        if super::is_speech_with_detector(chunk, detector, super::VAD_THRESHOLD) {
            speech.extend_from_slice(chunk);
        }
    }
    speech
}

/// Generate owner-negative embeddings (non-wake-word TTS phrases, re-scoped).
///
/// These are TTS-synthesised phrases from the ENROLLED voice only
/// (single-speaker semantics — the owner's own non-wake speech); the
/// multi-voice style rotation is gone.  Documented limitation: TTS speech
/// cannot match the distribution of real human Phase 3 speech.
///
/// ## Alignment with the encoder pipeline
///
/// Production captures owner negatives as real mic audio (Phase 3 collection)
/// and encodes them through the shared Qwen3-ASR encoder
/// ([`crate::audio::wake_word::encode_window`]).  The benchmark's TTS
/// surrogates are therefore VAD-gated through a dedicated earshot detector
/// (per-clip fresh, mirroring the streaming path's per-segment detector) and
/// the VAD-positive speech is encoded once.  A bounded low-SNR cell (brown
/// noise at 10 dB on the VAD-gated speech) is emitted per phrase × seed — the
/// enrolled voice under noise must not match the prototype.
///
/// Returns one 1024-dim L2-normalized embedding per accepted cell.
fn generate_owner_negative_sequences(
    enrolled_style: &str,
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Vec<Vec<f32>> {
    let Some(model) = crate::audio::local_transcriber::shared_model_arc() else {
        warn!("Owner negatives: shared Qwen3-ASR model not loaded");
        return Vec::new();
    };
    let mut embeddings = Vec::new();
    for (i, &phrase) in OWNER_NEGATIVE_PHRASES.iter().enumerate() {
        for seed in 0..3 {
            // Fresh collision-free range (9000+) — the old 1000-1014 range
            // overlapped the confusable base (1000+).
            let seed_val = 9000 + i as u64 * 3 + seed as u64;
            if let Some(pcm) = synthesize_with_pcm_cache(
                phrase,
                enrolled_style,
                seed_val,
                TARGET_SAMPLE_RATE,
                model_hash,
                cache_dir,
            ) {
                // VAD-gate with a dedicated detector (never the global
                // VAD_DETECTOR) — only VAD-positive hops are encoded.
                let mut detector = Detector::default();
                let speech_audio = vad_gate_speech(&pcm, &mut detector);
                if speech_audio.is_empty() {
                    warn!(
                        "Owner-negative '{phrase}' seed {seed}: no VAD-positive \
                         speech — skipping"
                    );
                    continue;
                }
                match crate::audio::wake_word::encode_window(&model, &speech_audio) {
                    Ok(emb) => embeddings.push(emb),
                    Err(e) => warn!("Owner-negative '{phrase}' seed {seed}: encode failed: {e}"),
                }
                // Bounded low-SNR cell: brown noise at 10 dB on the VAD-gated
                // speech — the enrolled voice under noise must not trigger.
                let noisy = crate::util::add_noise_color(
                    &speech_audio,
                    10.0,
                    crate::util::NoiseColor::Brown,
                    seed_val,
                );
                match crate::audio::wake_word::encode_window(&model, &noisy) {
                    Ok(emb) => embeddings.push(emb),
                    Err(e) => {
                        warn!("Owner-negative '{phrase}' seed {seed}: brown-10 encode failed: {e}");
                    }
                }
            }
        }
    }
    embeddings
}

/// Generate ambient noise embeddings for negative calibration.
///
/// Produces one 1024-dim L2-normalized window embedding per noise profile × 2
/// SNR levels (full amplitude and −6 dB).
///
/// ## Alignment with the encoder pipeline
///
/// Production captures real ambient negatives as raw mic audio collected
/// during Phase 3 and encodes them through the shared Qwen3-ASR encoder.  The
/// benchmark's synthetic noise profiles are encoded directly (one 1 s window
/// per profile × 2 levels) via [`crate::audio::wake_word::encode_window`].
/// Documented limitation: synthetic noise is a surrogate for real ambient
/// capture.
fn generate_ambient_noise_sequences() -> Vec<Vec<f32>> {
    let Some(model) = crate::audio::local_transcriber::shared_model_arc() else {
        warn!("Ambient negatives: shared Qwen3-ASR model not loaded");
        return Vec::new();
    };
    let mut embeddings = Vec::new();
    for (label, noise_fn) in NOISE_PROFILES {
        let raw = noise_fn();
        // Level 1: full amplitude
        match crate::audio::wake_word::encode_window(&model, &raw) {
            Ok(emb) => embeddings.push(emb),
            Err(e) => warn!("Ambient '{label}' level-0: encode failed: {e}"),
        }
        // Level 2: reduced amplitude (-6dB)
        let attenuated = crate::util::apply_gain(&raw, -6.0);
        match crate::audio::wake_word::encode_window(&model, &attenuated) {
            Ok(emb) => embeddings.push(emb),
            Err(e) => warn!("Ambient '{label}' level-1: encode failed: {e}"),
        }
    }
    embeddings
}

/// Bench-local restricted-style confusable/unrelated negative embeddings.
///
/// Production's old prewarm rotated ALL 10 TTS voices into the training
/// negatives — the bench restricts the style list so voices outside the
/// negative pool (M2-M5) are absent from EVERY training path.  This
/// replicates the encoder pipeline's negative recipe over the restricted
/// style list: synthesize via the shared PCM cache, VAD-gate through a
/// dedicated earshot detector, encode the VAD-positive speech via
/// [`crate::audio::wake_word::encode_window`].
///
/// There is no embedding-level disk cache; the bench pays the encoder forwards
/// on each run — the PCM cache keeps TTS synthesis cheap.
fn generate_restricted_phrase_negatives(
    phrase_type: &'static str,
    phrases: &'static [&'static str],
    seeds_per_phrase: usize,
    seed_base: u64,
    styles: &[String],
    model_hash: &str,
    cache_dir: &std::path::Path,
) -> Vec<Vec<f32>> {
    let Some(model) = crate::audio::local_transcriber::shared_model_arc() else {
        warn!("{phrase_type} negatives: shared Qwen3-ASR model not loaded");
        return Vec::new();
    };
    if styles.is_empty() {
        // No negative-pool styles (fewer-than-6 TTS voices) — the call-site
        // non-empty asserts below fail loudly rather than calibrating a weaker
        // enrollment.
        return Vec::new();
    }
    let num_styles = styles.len();
    let mut embeddings: Vec<Vec<f32>> = Vec::new();

    for (i, &phrase) in phrases.iter().enumerate() {
        for seed_idx in 0..seeds_per_phrase {
            // Same round-robin style formula as production, restricted to the
            // negative-pool styles.
            let style_idx = (i * seeds_per_phrase + seed_idx) % num_styles;
            let style = &styles[style_idx];
            let seed = seed_base + i as u64 * seeds_per_phrase as u64 + seed_idx as u64;

            // PCM from cache or fresh TTS synthesis.
            let Some(pcm) = synthesize_with_pcm_cache(
                phrase,
                style,
                seed,
                TARGET_SAMPLE_RATE,
                model_hash,
                cache_dir,
            ) else {
                continue;
            };

            // VAD-gate with a dedicated detector (never the global
            // VAD_DETECTOR) — only VAD-positive speech is encoded.
            let mut detector = Detector::default();
            let speech_audio = vad_gate_speech(&pcm, &mut detector);
            if speech_audio.is_empty() {
                warn!(
                    "{phrase_type} phrase '{phrase}' (seed {seed}) produced no \
                     VAD-positive speech — skipping (matches streaming: no \
                     speech ⇒ no embeddings)"
                );
                continue;
            }
            match crate::audio::wake_word::encode_window(&model, &speech_audio) {
                Ok(emb) => embeddings.push(emb),
                Err(e) => {
                    warn!("{phrase_type} phrase '{phrase}' (seed {seed}): encode failed: {e}");
                }
            }
        }
    }
    embeddings
}

/// Compute VAD frame decisions and segment audio into utterances at the
/// enrollment VAD threshold, using a fresh earshot detector (never the global
/// `VAD_DETECTOR`).
///
/// Each frame (stride = [`HOP_LENGTH`](super::HOP_LENGTH)) feeds only its NEW
/// [`HOP_LENGTH`](super::HOP_LENGTH) samples to the detector — the production
/// feed contract — and the decisions are passed to
/// [`super::segment_utterances_by_vad`] with the standard pipeline config.
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
        // to maintain train-inference consistency.
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

/// Process enrollment clips through VAD-gated utterance segmentation and
/// encode each utterance through the shared Qwen3-ASR encoder.
///
/// The encoder pipeline has no trainable head and no AGC/NS preprocessing:
/// each raw TTS clip is VAD-segmented (fresh earshot detector per clip via
/// [`compute_vad_segments`]) and every utterance is embedded once via
/// [`crate::audio::wake_word::encode_window`], producing one 1024-dim
/// L2-normalized embedding per utterance.
///
/// Returns `Vec<Vec<f32>>` — one embedding per VAD utterance.
fn vad_segment_and_enroll(enrollment_variants: &[(Vec<f32>, String)]) -> Vec<Vec<f32>> {
    const SILENCE_GAP_SAMPLES: usize = 2 * 16_000; // 2.0 s at 16 kHz

    let Some(model) = crate::audio::local_transcriber::shared_model_arc() else {
        panic!("FATAL: shared Qwen3-ASR model not loaded — cannot run VAD-gated enrollment");
    };

    // Concatenate all clips with 2.0 s digital-silence gaps, then VAD-segment
    // the combined stream.  `segment_utterances_by_vad` only emits an
    // utterance once it sees `ENROLLMENT_SILENCE_THRESHOLD_SAMPLES` (~304 ms)
    // of trailing silence — a lone clip ending mid-speech never terminates.
    // The concatenation carries no preprocessing (the encoder pipeline
    // consumes raw audio) so every clip yields exactly one utterance.
    let silence: Vec<f32> = vec![0.0f32; SILENCE_GAP_SAMPLES];
    let mut combined: Vec<f32> = Vec::new();
    for (pcm, _label) in enrollment_variants {
        if !combined.is_empty() {
            combined.extend_from_slice(&silence);
        }
        combined.extend_from_slice(pcm);
    }
    combined.extend_from_slice(&silence);

    let (_vad_decisions, utterances) = compute_vad_segments(&combined);
    info!(
        "VAD concatenation: {} total samples ({:.1}s) from {} originals with 2.0s gaps",
        combined.len(),
        combined.len() as f64 / f64::from(super::SAMPLE_RATE),
        enrollment_variants.len(),
    );
    info!(
        "VAD segmentation: {} utterances from {} concatenated originals",
        utterances.len(),
        enrollment_variants.len(),
    );

    let mut utterance_embeddings: Vec<Vec<f32>> = Vec::new();
    for (i, utterance) in utterances.iter().enumerate() {
        match crate::audio::wake_word::encode_window(&model, utterance) {
            Ok(emb) => utterance_embeddings.push(emb),
            Err(e) => warn!("Enrollment utterance {i}: encode failed: {e}"),
        }
    }

    info!(
        "VAD-gated enrollment: {} utterance embeddings from {} VAD utterances across {} clips \
         (encoder pipeline — one embedding per utterance, no augmentation)",
        utterance_embeddings.len(),
        utterances.len(),
        enrollment_variants.len(),
    );

    utterance_embeddings
}

// ── Streaming detection ─────────────────────────────

/// Result from [`run_streaming_detection`].
struct DetectionResult {
    detected: bool,
    /// Adaptive threshold state captured at the end of the test utterance
    /// (before the trailing silence flush).  A segment boundary fired during
    /// the flush calls `reset_detection_segment`, which resets
    /// `ctx.adaptive_threshold` to bootstrap; carrying this pre-flush snapshot
    /// forward lets the benchmark's shared-state design preserve
    /// the adaptation accumulated from the utterance's frames instead of
    /// re-bootstrapping per variant.
    adaptive_state_pre_flush: super::AdaptiveThresholdState,
}

/// Run the production streaming wake word detection pipeline on audio samples.
///
/// Feeds audio through [`handle_wake_word_detection`] in FRAME_LENGTH chunks,
/// exercising the full streaming chain: raw accumulation, VAD gating,
/// stride-gated window encoding, [`score_single_embedding`], and cooldown
/// logic.
///
/// After all audio is fed, silence frames are sent to flush any remaining
/// detection state (matching how the production pipeline handles
/// speech→silence transitions).
///
/// Returns a [`DetectionResult`] with the detection flag.
fn run_streaming_detection(samples: &[f32], ctx: &mut super::PipelineCtx) -> DetectionResult {
    // Save pre-existing timestamp — we only return true if detection fires
    // during THIS call, not because a prior call already set the field.
    let before = ctx.last_wake_word_detection;

    // Feed audio in FRAME_LENGTH chunks through the raw-sample streaming path
    // (no AGC/NS — the encoder pipeline consumes raw audio directly).
    for chunk in samples.chunks(super::FRAME_LENGTH) {
        process_frame(chunk, ctx);
        if ctx.last_wake_word_detection != before {
            // No flush ran on this path, so the pre-flush snapshot is the
            // current state; the boundary-fire branch in the callers is
            // unreachable when `detected` is true anyway.
            return DetectionResult {
                detected: true,
                adaptive_state_pre_flush: ctx.adaptive_threshold.clone(),
            };
        }
    }

    // ── Trailing silence flush ───────────────────────────
    // Production continues on natural silence after the utterance and
    // detection can still fire for up to SEGMENT_TIMEOUT_HOPS (19) VAD-negative
    // hops before the segment-boundary reset clears the per-segment state.
    // The old flush fed exactly 3×512 zero frames (~96 ms), which
    // missed wake words whose rolling sum crossed threshold during the longer
    // production window (~304 ms).
    //
    // The adaptive threshold state is captured BEFORE the flush: a segment
    // boundary fired during the flush resets it to bootstrap, and the
    // benchmark's shared-state design wants the adapted state to survive the
    // boundary rather than re-bootstrapping per variant.
    let adaptive_state_pre_flush = ctx.adaptive_threshold.clone();
    //
    // Feed FRAME_LENGTH digital-silence chunks (production's mic delivers real
    // zeros, not empty Vecs — see process_frame) until:
    //   1. detection fires, or
    //   2. the segment boundary fires, or
    //   3. we exceed the production window (SEGMENT_TIMEOUT_HOPS hops plus a
    //      small allowance for any remaining VAD-positive utterance tail).
    //
    // The boundary is detected via the rolling score window:
    // reset_detection_segment clears it (and the stride-gated scorer also
    // clears it via NO_MATCH_RESET when a trailing-silence window scores
    // below the reset threshold), so an empty window means no further
    // detection is possible — further silence cannot produce a detection.
    let mut silence_chunks = 0usize;
    while silence_chunks < super::SEGMENT_TIMEOUT_HOPS + 4 {
        if ctx.last_wake_word_detection != before {
            return DetectionResult {
                detected: true,
                adaptive_state_pre_flush,
            };
        }
        let window_was_nonempty = !ctx.score_window.is_empty();
        process_frame(&vec![0.0; super::FRAME_LENGTH], ctx);
        silence_chunks += 1;
        // Segment boundary (or NO_MATCH reset) emptied the rolling window —
        // further silence cannot produce a detection, so stop feeding.
        if window_was_nonempty && ctx.score_window.is_empty() {
            break;
        }
    }

    DetectionResult {
        detected: ctx.last_wake_word_detection != before,
        adaptive_state_pre_flush,
    }
}

/// Track false reactions on a set of negative variants: every detection on
/// the set is pushed to `false_accepts` (the wake word must NOT fire on the
/// non-phrase corpus).
#[derive(Debug, Default)]
struct DetectionMetrics {
    false_accepts: Vec<String>,
}

/// Process a list of audio clips through the detection pipeline, recording
/// results in `metrics`.  Shared helper for the negative detection blocks,
/// eliminating the repetitive match-and-track boilerplate.
///
/// # Parameters
/// - `variants`: audio clips with descriptive labels.
/// - `metrics`: `on_detection` fills `.false_accepts`.
/// - `on_detection`: called with `(&mut metrics, label_str)` when the
///   wake word is detected; negatives push to `.false_accepts`.
fn test_detection_samples(
    variants: &[(Vec<f32>, String)],
    metrics: &mut DetectionMetrics,
    on_detection: impl Fn(&mut DetectionMetrics, &str),
    mut adaptive_state: Option<&mut super::AdaptiveThresholdState>,
    cold_start: bool,
) {
    // The enrollment is set in global state ONCE before the detection metrics
    // (via super::set_enrollment) — handle_wake_word_detection reads it from
    // voice_state() per scoring step.

    for (i, (samples, label)) in variants.iter().enumerate() {
        info!(
            "  Variant {}/{}: {label} — processing ({})",
            i + 1,
            variants.len(),
            if cold_start { "cold start" } else { "warm" }
        );
        let mut ctx = super::PipelineCtx::new();
        // Clone the shared adaptive state into this variant's ctx so the
        // adaptive threshold is active from the first frame, simulating a
        // continuous pipeline.  Without this the
        // adaptive state never exits its 5-frame bootstrap because each
        // variant gets a fresh ctx, keeping all benchmark metrics measured
        // against the static threshold.
        //
        // Cold start: production bootstraps the adaptive
        // threshold from the first 5 real per-frame scores of actual background
        // audio after each segment boundary.  The cold pass therefore uses a
        // fresh AdaptiveThresholdState::new() per variant (no shared-state
        // propagation) so the threshold trajectory matches production exactly.
        if !cold_start && let Some(ref mut state) = adaptive_state {
            ctx.adaptive_threshold = state.clone();
        }
        // Warm pass: feed warm-up audio so the ring is primed and the
        // adaptive path is warm before the utterance.  The cold pass skips
        // consume_warmup — production has no warm-up after a silence boundary.
        if !cold_start {
            consume_warmup(&mut ctx);
        }
        let result = run_streaming_detection(samples, &mut ctx);
        // Propagate the updated adaptive state for the next variant (warm pass
        // only — the cold pass keeps each variant's bootstrap independent).
        //
        // When the trailing silence fires a segment boundary (`!detected` AND
        // the rolling window was cleared — reset_detection_segment is the
        // non-detection path that empties the window), the boundary
        // resets ctx.adaptive_threshold back to bootstrap.  Propagating that
        // bootstrap state would defeat the shared-state design
        // which keeps the adaptive code path active across variants to avoid
        // measuring everything against the static threshold — and with the
        // extended flush every non-detecting variant fires the boundary.
        // Instead, carry the pre-flush snapshot captured at the end of the
        // utterance so the shared state keeps adapting to each variant's
        // frames.  The cold pass measures the bootstrap behavior separately.
        if !cold_start && let Some(ref mut state) = adaptive_state {
            let boundary_fired = !result.detected && ctx.score_window.is_empty();
            if boundary_fired {
                **state = result.adaptive_state_pre_flush.clone();
            } else {
                **state = ctx.adaptive_threshold.clone();
            }
        }
        let peak = ctx.instrumentation.peak_score;
        if result.detected {
            on_detection(metrics, label);
        }
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

/// The full negative detection corpus, built ONCE by [`build_negative_corpus`]
/// and consumed by the false-reaction metric (confusable bands 800/810,
/// unrelated bands 900/910, silence, noise profiles).
struct NegativeCorpus {
    /// Confusable band 800 + confusable2 band 810 (merged).
    confusable: Vec<(Vec<f32>, String)>,
    /// Unrelated band 900 + unrelated2 band 910 (merged).
    unrelated: Vec<(Vec<f32>, String)>,
    /// Silence profiles ([`SILENCE_DURATIONS`]).
    silence: Vec<(Vec<f32>, String)>,
    /// Noise profiles ([`all_noise_profiles`]).
    noise: Vec<(Vec<f32>, String)>,
}

/// Build the negative detection corpus with the same generators and seed
/// bands as the detection metric — the single source of truth for the
/// negative corpus.  Deterministic given the TTS PCM cache.
fn build_negative_corpus(
    available_styles: &[String],
    model_version_hash: &str,
    cache_dir_path: &std::path::Path,
) -> NegativeCorpus {
    let conf_seed = |band: u64, prefix: &str| {
        generate_phrase_variants_cached(
            CONFUSABLE_PHRASES,
            available_styles,
            SeedConfig {
                base_seed: band,
                num_variants: 1, // single seed per phrase (detection test)
                seed_variant: 0,
            },
            prefix,
            model_version_hash,
            cache_dir_path,
        )
    };
    let mut confusable = conf_seed(800, "confusable");
    confusable.extend(conf_seed(810, "confusable2"));
    let unrel_seed = |band: u64, prefix: &str| {
        generate_phrase_variants_cached(
            UNRELATED_PHRASES,
            available_styles,
            SeedConfig {
                base_seed: band,
                num_variants: 1,
                seed_variant: 0,
            },
            prefix,
            model_version_hash,
            cache_dir_path,
        )
    };
    let mut unrelated = unrel_seed(900, "unrelated");
    unrelated.extend(unrel_seed(910, "unrelated2"));
    let silence: Vec<(Vec<f32>, String)> = SILENCE_DURATIONS
        .iter()
        .map(|(label, len)| (vec![0.0f32; *len], label.to_string()))
        .collect();
    let noise: Vec<(Vec<f32>, String)> = all_noise_profiles()
        .map(|(label, generator)| (generator(), (*label).to_string()))
        .collect();
    NegativeCorpus {
        confusable,
        unrelated,
        silence,
        noise,
    }
}

/// Run ONE enrolled-speaker clip through the real streaming cold pass and
/// report whether the wake word fired.
///
/// Cold pass: fresh [`PipelineCtx`] per clip, no warm-up, fresh adaptive
/// bootstrap — the production post-silence start, driven for real through
/// `handle_wake_word_detection`.
fn run_enrolled_cold_variant(pcm: &[f32]) -> bool {
    // Cold pass: fresh PipelineCtx per variant, no consume_warmup, fresh
    // AdaptiveThresholdState (matches production's post-silence start).
    let mut ctx = super::PipelineCtx::new();
    let result = run_streaming_detection(pcm, &mut ctx);
    result.detected
}

/// SHA-256 of a file (hex).
fn faph_sha256_file(path: &std::path::Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    let data = std::fs::read(path)?;
    Ok(crate::util::hex_string(&Sha256::digest(&data)))
}

/// Self-contained single-file download with SHA-256 verification.
///
/// Downloads `url` to `dest` (creating parent dirs), verifies the SHA-256
/// against the manifest pin, and only then writes the file.  Uses its own
/// local current-thread tokio runtime — deliberately NOT the embedder.rs
/// download precedent (gated embedder/providers out of the bench
/// surface), and no new runtime dependency for the standard bench run.
fn faph_download_file(
    url: &str,
    dest: &std::path::Path,
    expected_sha256: &str,
) -> Result<(), String> {
    use sha2::{Digest, Sha256};
    // Self-contained runtime: safe from a spawn_blocking context (no outer
    // async context is entered; Handle::try_current is checked first so a
    // runtime already present on this thread is reused rather than nested).
    let fut = async {
        crate::util::http::install_ring_provider();
        let client = reqwest::Client::new();
        let resp = client
            .get(url)
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("HTTP status {}", resp.status()));
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| format!("body read failed: {e}"))?;
        Ok::<Vec<u8>, String>(bytes.to_vec())
    };
    let bytes = if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.block_on(fut)
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("cannot build download runtime: {e}"))?;
        rt.block_on(fut)
    }?;
    let sha = crate::util::hex_string(&Sha256::digest(&bytes));
    if sha != expected_sha256 {
        return Err(format!(
            "SHA-256 mismatch: expected {expected_sha256}, got {sha}"
        ));
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
    }
    std::fs::write(dest, &bytes).map_err(|e| format!("write: {e}"))
}

/// Count raw false-accept events while feeding one corpus file as ambient
/// audio through a SHARED continuous-listening pipeline context (one pipeline
/// context across the whole corpus, matching production's continuous
/// listening).
///
/// Feeds every sample in [`FRAME_LENGTH`](super::FRAME_LENGTH) chunks through
/// [`handle_wake_word_detection`] directly, recording the
/// AUDIO position (seconds) of each fresh `last_wake_word_detection`
/// timestamp.  Production's `WAKE_WORD_COOLDOWN` gate inside
/// `handle_wake_word_detection` is wall-clock based: at the bench's
/// faster-than-real-time feed speed it would suppress far more audio than the
/// 3 s it represents, structurally
/// under-counting.  The bench therefore observes the RAW event stream by
/// re-arming after each event exactly like production's post-command reset
/// (`reset_pipeline_state(Soft)` + `is_recording = false`) AND clearing the
/// cooldown timestamp (bench-side emulation — production's gate is NOT
/// modified).  The caller applies production's 3 s cooldown as an
/// audio-position merge ([`faph_merge_events`]) for the
/// production-equivalent count.
///
/// `audio_pos` is the running audio position (seconds) across the whole
/// corpus; it advances as samples are fed so event positions are globally
/// comparable for the merge.
fn faph_feed_file_continuous(
    samples: &[f32],
    ctx: &mut super::PipelineCtx,
    audio_pos: &mut f64,
) -> Vec<f64> {
    let mut raw_events: Vec<f64> = Vec::new();
    for chunk in samples.chunks(super::FRAME_LENGTH) {
        process_frame(chunk, ctx);
        if ctx.last_wake_word_detection.is_some() {
            raw_events.push(*audio_pos);
            // Re-arm (production post-command reset): Soft preserves the
            // cooldown timestamp, so also clear it — the bench observes the
            // raw event stream; the 3 s cooldown is applied as an
            // audio-position merge by the caller.
            ctx.reset_pipeline_state(super::ResetLevel::Soft);
            ctx.is_recording = false;
            ctx.last_wake_word_detection = None;
        }
        *audio_pos += chunk.len() as f64 / f64::from(super::SAMPLE_RATE);
    }
    raw_events
}

/// Merge raw event audio positions under production's cooldown semantics.
///
/// After a detection, production suppresses re-detections within
/// [`WAKE_WORD_COOLDOWN`](super::WAKE_WORD_COOLDOWN) — 3 s of wall time,
/// which at 1× realtime equals 3 s of audio.  Events whose audio positions
/// are less than that apart are merged (the later one would have been
/// suppressed).  `events` must be sorted ascending by audio position (the
/// feed loop produces them in order).
fn faph_merge_events(events: &[f64], cooldown_secs: f64) -> usize {
    let mut merged = 0usize;
    let mut last_kept: Option<f64> = None;
    for &pos in events {
        match last_kept {
            Some(prev) if pos - prev < cooldown_secs => {}
            _ => {
                merged += 1;
                last_kept = Some(pos);
            }
        }
    }
    merged
}

/// Clear the session-lifetime instrumentation between corpus files in the
/// continuous-listening real-audio feed.
///
/// The real-audio phase only counts detection events and the VAD-active
/// frames — [`DetectionInstrumentation`] holds just `vad_speech_frames` and
/// `peak_score` (the per-variant diagnostic fields were pruned with the old
/// e2e bench), and the VAD counter must reset per file+gap window so the
/// FA/h denominator is window-scoped.  The acoustic state the continuous
/// feed depends on (audio ring, adaptive threshold, VAD detector) is
/// preserved.
fn faph_clear_instrumentation(ctx: &mut super::PipelineCtx) {
    ctx.instrumentation.vad_speech_frames = 0;
}

// ═══════════════════════════════════════════════════════════════════════
// Wake-word benchmark baseline: three plain metrics
// ═══════════════════════════════════════════════════════════════════════
//
// The whole report is exactly three metrics:
//   1. Recognition — X of 40 phrase utterances recognized (fixed bench
//      phrase BENCH_WAKE_PHRASE, the existing 40-clip held-out basis).
//   2. False reactions — N on the 113 non-phrase set + a rate per hour on
//      real audio (parallel feed of the pinned subset below).
//   3. Data coverage — 40 utterances + 113 non-phrases + the real-audio
//      hours (speech + noise) + run wall time.
// plus the worker count and the FA/h basis note.  No per-frame arrays, no
// analysis sections, no old-run comparisons.

/// Pinned real-audio subset for the wake-word bench's FA-per-hour metric.
///
/// Selection heuristic: the longest librivox
/// speech files up to a target + ALL us-gov speech files + a fixed noise
/// selection (the longest free-sound and sound-bible clips), covering both
/// false-reaction regimes (spontaneous confusables in continuous speech;
/// noise-triggered triggers).  The result is PINNED — paths in fixed order,
/// so the file order, the round-robin worker assignment, and therefore the
/// merged outcome are run-to-run reproducible at the outcome level.
///
/// Measured durations at pin time (16 kHz mono WAV): the 6 longest librivox
/// files total ≈ 31.2 min, the 12 us-gov files ≈ 25.0 min, the 10
/// free-sound clips ≈ 5.3 min, and the 5 sound-bible clips ≈ 2.4 min — 33
/// files, ≈ 63.9 min ≈ 1.07 h (0.94 h speech + 0.13 h noise).
///
/// Budget note: the ≤10-minute cap applies to the whole warm
/// run.  The parallel feed is memory-bandwidth-bound on the encoder (8
/// workers measured ≈ 1.5× serial throughput on the M3 Pro validation
/// machine), so the first pinned list (~2.63 h) exceeded the cap and was
/// shrunk to this list; the report discloses the final coverage.
const BENCH_REAL_AUDIO_SUBSET: &[&str] = &[
    // 6 longest librivox speech files (≈31.2 min)
    "speech/librivox/speech-librivox-0027.wav",
    "speech/librivox/speech-librivox-0051.wav",
    "speech/librivox/speech-librivox-0093.wav",
    "speech/librivox/speech-librivox-0075.wav",
    "speech/librivox/speech-librivox-0103.wav",
    "speech/librivox/speech-librivox-0088.wav",
    // All 12 us-gov speech files (≈25.0 min)
    "speech/us-gov/speech-us-gov-0252.wav",
    "speech/us-gov/speech-us-gov-0147.wav",
    "speech/us-gov/speech-us-gov-0082.wav",
    "speech/us-gov/speech-us-gov-0159.wav",
    "speech/us-gov/speech-us-gov-0067.wav",
    "speech/us-gov/speech-us-gov-0250.wav",
    "speech/us-gov/speech-us-gov-0210.wav",
    "speech/us-gov/speech-us-gov-0229.wav",
    "speech/us-gov/speech-us-gov-0201.wav",
    "speech/us-gov/speech-us-gov-0130.wav",
    "speech/us-gov/speech-us-gov-0005.wav",
    "speech/us-gov/speech-us-gov-0109.wav",
    // 10 longest free-sound noise clips (≈5.3 min)
    "noise/free-sound/noise-free-sound-0640.wav",
    "noise/free-sound/noise-free-sound-0184.wav",
    "noise/free-sound/noise-free-sound-0761.wav",
    "noise/free-sound/noise-free-sound-0012.wav",
    "noise/free-sound/noise-free-sound-0605.wav",
    "noise/free-sound/noise-free-sound-0298.wav",
    "noise/free-sound/noise-free-sound-0251.wav",
    "noise/free-sound/noise-free-sound-0733.wav",
    "noise/free-sound/noise-free-sound-0035.wav",
    "noise/free-sound/noise-free-sound-0709.wav",
    // 5 longest sound-bible noise clips (≈2.4 min)
    "noise/sound-bible/noise-sound-bible-0075.wav",
    "noise/sound-bible/noise-sound-bible-0044.wav",
    "noise/sound-bible/noise-sound-bible-0067.wav",
    "noise/sound-bible/noise-sound-bible-0014.wav",
    "noise/sound-bible/noise-sound-bible-0032.wav",
];

/// Worker count for the wake-word bench's parallel real-audio feed — a
/// pinned bench constant (NOT an env knob).
const BENCH_WORKERS: usize = 8;

/// File→worker assignment shape for the parallel real-audio feed: round-robin
/// over [`BENCH_REAL_AUDIO_SUBSET`] in pinned order (worker `i` feeds
/// files `i`, `i + BENCH_WORKERS`, `i + 2·BENCH_WORKERS`, …).
/// Pinned as a bench constant — deterministic per run.
const BENCH_ASSIGNMENT: &str = "round_robin";

/// Inter-file silence gap for each worker's continuous-listening stream
/// (2 s at 16 kHz, so the natural segment-boundary reset fires between
/// files).
const BENCH_FILE_GAP_SAMPLES: usize = 32_000;

/// Per-worker real-audio feed totals for the wake-word bench.
///
/// Cooldown merge applies PER WORKER: events on different workers are
/// independent continuous-listening streams and are never merged across
/// workers.  Totals are summed across workers afterwards.
#[derive(Default)]
struct WorkerTotals {
    files_fed: u64,
    audio_secs: f64,
    speech_audio_secs: f64,
    noise_audio_secs: f64,
    vad_active_secs: f64,
    raw_events: Vec<f64>,
    merged_events: usize,
}

impl WorkerTotals {
    fn merge(&mut self, other: WorkerTotals) {
        self.files_fed += other.files_fed;
        self.audio_secs += other.audio_secs;
        self.speech_audio_secs += other.speech_audio_secs;
        self.noise_audio_secs += other.noise_audio_secs;
        self.vad_active_secs += other.vad_active_secs;
        self.raw_events.extend(other.raw_events);
        self.merged_events += other.merged_events;
    }
}

/// Feed one worker's assigned real-audio files through the SHARED production
/// detection path — `handle_wake_word_detection` via [`faph_feed_file_continuous`]
/// — in a per-worker continuous-listening [`PipelineCtx`] with its OWN
/// injected VAD detector (`ctx.injected_vad`, the voice-tests seam).  The
/// global [`VAD_DETECTOR`](super::VAD_DETECTOR) singleton is never touched by
/// parallel workers, so the streams stay independent.  Detection LOGIC is
/// unchanged — only the detector instance differs.
fn worker_feed(files: &[(String, String, u64)], cache_root: &std::path::Path) -> WorkerTotals {
    const GAP_SAMPLES: usize = BENCH_FILE_GAP_SAMPLES;
    let gap: Vec<f32> = vec![0.0; GAP_SAMPLES];
    let mut ctx = super::PipelineCtx::new();
    // Per-worker detector instance (the voice-tests seam).  Preserved across
    // the Soft resets the feed performs after each event (see the field's
    // doc comment in voice.rs).
    ctx.injected_vad = Some(earshot::Detector::default());
    let mut audio_pos = 0.0f64;
    let mut totals = WorkerTotals::default();
    for (path, _sha256, _size) in files {
        let p = cache_root.join(path);
        let samples = match crate::audio::local_transcriber::decode_audio_to_mono_f32(&p) {
            Ok(s) => s,
            Err(e) => {
                warn!("wake-word bench: decode failed for {path}: {e}");
                continue;
            }
        };
        let audio_secs = samples.len() as f64 / f64::from(super::SAMPLE_RATE);
        totals.audio_secs += audio_secs;
        if path.starts_with("speech/") {
            totals.speech_audio_secs += audio_secs;
        } else {
            totals.noise_audio_secs += audio_secs;
        }
        totals.raw_events.extend(faph_feed_file_continuous(
            &samples,
            &mut ctx,
            &mut audio_pos,
        ));
        totals.files_fed += 1;
        // Inter-file silence gap: fires the natural segment-boundary reset
        // (fresh detector state, adaptive bootstrap, cleared ring) —
        // continuous-listening semantics.
        totals
            .raw_events
            .extend(faph_feed_file_continuous(&gap, &mut ctx, &mut audio_pos));
        // VAD-active seconds for the file+gap window from the CONTINUOUS feed
        // (per-worker detector): the FA/h denominator basis (the gap is
        // digital silence, ~0 contribution).
        totals.vad_active_secs += ctx.instrumentation.vad_speech_frames as f64
            * super::HOP_LENGTH as f64
            / f64::from(super::SAMPLE_RATE);
        // Bound memory: drop the per-file diagnostic vectors (never read in
        // this phase) while preserving the continuous acoustic state.
        faph_clear_instrumentation(&mut ctx);
    }
    totals.merged_events =
        faph_merge_events(&totals.raw_events, super::WAKE_WORD_COOLDOWN.as_secs_f64());
    totals
}

/// Build a documented-skip report for the wake-word bench's real-audio phase
/// (degraded-skip contract — a loud 'not measured' marker, never a silent
/// drop).
fn skip_json(reason_key: &str, detail: &str) -> serde_json::Value {
    warn!("Wake-word bench real-audio phase skipped: {reason_key} — {detail}");
    eprintln!("         Wake-word bench real-audio phase skipped: {reason_key} — {detail}");
    serde_json::json!({
        "status": "skipped",
        "metric": "FA rate per hour on real audio — NOT MEASURED (degraded skip)",
        "skip_reason": reason_key,
        "skip_detail": detail,
    })
}

/// Run the wake-word bench's real-audio phase: feed the pinned subset in
/// parallel ([`BENCH_WORKERS`] workers, round-robin file assignment),
/// each worker through its own [`PipelineCtx`] + injected detector, using the
/// SAME production detection path.
///
/// Always returns a report (ran or documented-skip — never a silent drop)
/// when the phase cannot run: corpus files are missing/unverifiable —
/// download-on-demand is attempted first, then a loud skip.
#[expect(clippy::too_many_lines)]
fn run_real_audio_phase() -> serde_json::Value {
    let phase_start = Instant::now();

    // ── Corpus manifest (pinned, embedded at compile time; the subset is the
    // pinned path list above). ──
    let manifest: serde_json::Value =
        match serde_json::from_str(include_str!("faph_corpus_manifest.json")) {
            Ok(m) => m,
            Err(e) => {
                return skip_json(
                    "manifest_parse_failed",
                    &format!("embedded manifest failed to parse: {e}"),
                );
            }
        };
    let manifest_files: Vec<(String, String, u64)> = match manifest["files"].as_array() {
        Some(arr) => arr
            .iter()
            .filter_map(|f| {
                Some((
                    f[0].as_str()?.to_string(),
                    f[1].as_str()?.to_string(),
                    f[2].as_u64()?,
                ))
            })
            .collect(),
        None => Vec::new(),
    };
    let repo = manifest["repo"].as_str().unwrap_or("alexwengg/musan_mini");
    let revision = manifest["revision"].as_str().unwrap_or("");
    if manifest_files.is_empty() {
        return skip_json("manifest_empty", "manifest file list is empty");
    }

    // ── Resolve the pinned subset against the manifest, preserving the
    // pinned order (run-to-run reproducible file order). ──
    let mut subset: Vec<(String, String, u64)> = Vec::with_capacity(BENCH_REAL_AUDIO_SUBSET.len());
    for &path in BENCH_REAL_AUDIO_SUBSET {
        match manifest_files.iter().find(|(p, _, _)| p == path) {
            Some((p, sha, size)) => subset.push((p.clone(), sha.clone(), *size)),
            None => {
                return skip_json(
                    "subset_path_not_in_manifest",
                    &format!("pinned subset path missing from manifest: {path}"),
                );
            }
        }
    }

    // ── Corpus cache root (self-contained, persistent, under ~/.mahbot/) ──
    let cache_root = match crate::config::default_config_dir() {
        Ok(d) => d.join("faph_corpus"),
        Err(e) => {
            return skip_json(
                "cache_root_unavailable",
                &format!("cannot resolve ~/.mahbot for corpus cache: {e}"),
            );
        }
    };

    // ── Self-contained download/verify (the existing FAPH contract) ──
    let mut download_errors: Vec<String> = Vec::new();
    for (path, sha256, size) in &subset {
        let dest = cache_root.join(path);
        if dest.exists() && std::fs::metadata(&dest).is_ok_and(|m| m.len() == *size) {
            match faph_sha256_file(&dest) {
                Ok(h) if h == *sha256 => continue,
                Ok(_) => {
                    download_errors.push(format!("{path}: hash mismatch on cached file"));
                    continue;
                }
                Err(e) => {
                    download_errors.push(format!("{path}: read error {e}"));
                    continue;
                }
            }
        }
        let url = format!("https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}");
        match faph_download_file(&url, &dest, sha256) {
            Ok(()) => {
                info!("wake-word bench corpus: downloaded {path} ({size} bytes)");
            }
            Err(e) => {
                download_errors.push(format!("{path}: {e}"));
            }
        }
    }
    if !download_errors.is_empty() {
        let reason = format!(
            "corpus subset incomplete — {} file(s) failed download/verify (first: {})",
            download_errors.len(),
            download_errors[0],
        );
        return skip_json("corpus_download_failed", &reason);
    }

    // ── Parallel feed: round-robin assignment over the pinned order ──
    let worker_files: Vec<Vec<(String, String, u64)>> = (0..BENCH_WORKERS)
        .map(|w| {
            subset
                .iter()
                .skip(w)
                .step_by(BENCH_WORKERS)
                .cloned()
                .collect()
        })
        .collect();
    let mut handles = Vec::with_capacity(BENCH_WORKERS);
    for wf in worker_files {
        let cr = cache_root.clone();
        handles.push(std::thread::spawn(move || worker_feed(&wf, &cr)));
    }
    let mut totals = WorkerTotals::default();
    let mut worker_panic: Option<String> = None;
    for handle in handles {
        match handle.join() {
            Ok(wt) => totals.merge(wt),
            Err(e) => {
                if worker_panic.is_none() {
                    worker_panic = Some(format!("a real-audio worker thread panicked: {e:?}"));
                }
            }
        }
    }
    // Join ALL workers even after a panic so no worker thread is left
    // detached while the skip report proceeds; then report the panic.
    if let Some(reason) = worker_panic {
        return skip_json("worker_panicked", &reason);
    }

    let wall_secs = phase_start.elapsed().as_secs_f64();
    let audio_hours = totals.audio_secs / 3600.0;
    let vad_active_hours = totals.vad_active_secs / 3600.0;
    let merged_events = totals.merged_events;
    let raw_events = totals.raw_events.len();
    let fa_per_hour_raw = if audio_hours > 0.0 {
        merged_events as f64 / audio_hours
    } else {
        f64::NAN
    };
    let fa_per_hour_vad = if vad_active_hours > 0.0 {
        merged_events as f64 / vad_active_hours
    } else {
        f64::NAN
    };

    info!(
        "Wake-word bench real audio: {files_fed} files fed, {audio_hours:.2} h audio \
         ({vad_active_hours:.2} h VAD-active), {merged_events} cooldown-merged FA events \
         (raw {raw_events}), {fa_per_hour_vad:.4} FA/h VAD-active, {wall_secs:.1}s wall \
         ({workers} parallel workers, {BENCH_ASSIGNMENT})",
        files_fed = totals.files_fed,
        workers = BENCH_WORKERS,
    );
    eprintln!(
        "         Wake-word bench real audio: {files_fed} files fed, {audio_hours:.2} h audio \
         ({vad_active_hours:.2} h VAD-active), {merged_events} cooldown-merged FA events \
         (raw {raw_events}), {fa_per_hour_vad:.4} FA/h VAD-active, {wall_secs:.1}s wall \
         ({workers} parallel workers, {BENCH_ASSIGNMENT})",
        files_fed = totals.files_fed,
        workers = BENCH_WORKERS,
    );

    serde_json::json!({
        "status": "ran",
        "metric": "SPONTANEOUS-CONFUSABLE FA rate on real audio — the pinned subset \
                   contains ~zero wake-word utterances, so every detection is a \
                   spontaneous false accept",
        "subset": {
            "files_total": BENCH_REAL_AUDIO_SUBSET.len(),
            "speech_files": BENCH_REAL_AUDIO_SUBSET
                .iter()
                .filter(|p| p.starts_with("speech/"))
                .count(),
            "noise_files": BENCH_REAL_AUDIO_SUBSET
                .iter()
                .filter(|p| p.starts_with("noise/"))
                .count(),
            "selection_heuristic": "longest librivox speech files up to a target + ALL \
                                    us-gov speech files + a fixed noise selection (longest \
                                    free-sound + sound-bible clips) — pinned as a bench \
                                    constant",
        },
        "feed": {
            "workers": BENCH_WORKERS,
            "assignment": BENCH_ASSIGNMENT,
            "files_fed": totals.files_fed,
            "audio_hours_fed": audio_hours,
            "speech_hours_fed": totals.speech_audio_secs / 3600.0,
            "noise_hours_fed": totals.noise_audio_secs / 3600.0,
            "vad_active_hours_fed": vad_active_hours,
            "wall_secs": wall_secs,
        },
        "fa": {
            "raw_events": raw_events,
            "cooldown_merged_events": merged_events,
            "fa_per_hour_raw_audio": fa_per_hour_raw,
            "fa_per_hour_vad_active": fa_per_hour_vad,
            "basis_note": "Denominators are PER-WORKER sums: raw audio hours and \
                           VAD-active hours, each accumulated per worker over its own \
                           continuous-listening stream, then summed across workers.  \
                           Cooldown merge applies PER WORKER (events on different \
                           workers are independent streams and are never merged across \
                           workers); the number is a per-worker parallel-stream figure \
                           over the fixed pinned subset.",
        },
    })
}

/// Run the wake-word benchmark (three plain metrics).
///
/// Entry point invoked by `voice::run_wake_word_benchmark()` from
/// `benches/wake_word.rs`.
#[expect(clippy::cast_precision_loss, clippy::too_many_lines)]
pub(crate) fn run_wake_word_benchmark() {
    // ── Heartbeat drop guard ──────────────────────────────
    // A pulse every 60 s so the operator can confirm progress (the whole
    // warm run is capped at ~10 min).
    struct HeartbeatGuard(std::sync::Arc<std::sync::atomic::AtomicBool>);

    impl Drop for HeartbeatGuard {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .parse("info")
                .expect("info env filter"),
        )
        .try_init();

    let overall_start = Instant::now();
    info!("═══ Wake-Word Benchmark (three metrics) ═══");

    let heartbeat_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let _heartbeat_guard = HeartbeatGuard(heartbeat_stop.clone());
    let heartbeat_handle = {
        let stop = heartbeat_stop.clone();
        let start = overall_start;
        std::thread::spawn(move || {
            let mut counter: u64 = 0;
            loop {
                if counter.is_multiple_of(60) {
                    eprintln!(
                        "[heartbeat] wake_word benchmark still running — elapsed: {}m{:02}s",
                        start.elapsed().as_secs() / 60,
                        start.elapsed().as_secs() % 60,
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

    // ── 0. Initialize global state ──
    if crate::config::CONFIG.try_storage_root().is_none() {
        let mahbot_dir = crate::config::default_config_dir()
            .expect("Cannot resolve home directory for ~/.mahbot");
        crate::config::CONFIG.set_storage_root(mahbot_dir.clone());
        info!("CONFIG storage root set to: {}", mahbot_dir.display());
    }
    let cache_dir_path = cache_dir();
    if let Err(e) = std::fs::create_dir_all(&cache_dir_path) {
        eprintln!(
            "WARNING: Cannot create cache directory {}: {e}",
            cache_dir_path.display()
        );
    }
    let model_version_hash = tts_model_version_hash();
    info!("TTS model version hash: {}", &model_version_hash[..16]);
    crate::audio::tts::init_global()
        .unwrap_or_else(|e| warn!("tts::init_global() already called: {e}"));
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

    // ── Enrollment (shared machinery) ──
    let voice_allocation = allocate_voices(&available_styles);
    info!(
        "Voice allocation: enrolled={} negative_pool={:?}",
        voice_allocation.enrolled, voice_allocation.negative_pool,
    );
    let enrollment_variants = generate_enrollment_variants_cached(
        &voice_allocation.enrolled,
        &model_version_hash,
        &cache_dir_path,
    );
    if enrollment_variants.is_empty() {
        eprintln!(
            "FATAL: Need at least one enrollment variant. TTS synthesis may have failed for all styles."
        );
        return;
    }
    let train_clips = enrollment_variants;
    let utterance_embeddings = vad_segment_and_enroll(&train_clips);
    if utterance_embeddings.is_empty() {
        eprintln!(
            "FATAL: VAD-gated enrollment produced no utterances from {} training clips",
            train_clips.len(),
        );
        return;
    }
    info!(
        "VAD-gated enrollment: {} utterance embeddings from {} clips",
        utterance_embeddings.len(),
        train_clips.len(),
    );

    // Negative calibration pool (ambient → owner → confusable → unrelated).
    let confusable_neg_embeddings = generate_restricted_phrase_negatives(
        "confusable",
        CONFUSABLE_PHRASES,
        CONFUSABLE_SEEDS_PER_PHRASE,
        CONFUSABLE_SEED_BASE,
        &voice_allocation.negative_pool,
        &model_version_hash,
        &cache_dir_path,
    );
    let unrelated_neg_embeddings = generate_restricted_phrase_negatives(
        "unrelated",
        UNRELATED_PHRASES,
        UNRELATED_SEEDS_PER_PHRASE,
        UNRELATED_SEED_BASE,
        &voice_allocation.negative_pool,
        &model_version_hash,
        &cache_dir_path,
    );
    let ambient_neg_embeddings = generate_ambient_noise_sequences();
    let owner_neg_embeddings = generate_owner_negative_sequences(
        &voice_allocation.enrolled,
        &model_version_hash,
        &cache_dir_path,
    );
    let mut negative_embeddings: Vec<Vec<f32>> = Vec::new();
    negative_embeddings.extend(ambient_neg_embeddings);
    negative_embeddings.extend(owner_neg_embeddings);
    negative_embeddings.extend(confusable_neg_embeddings);
    negative_embeddings.extend(unrelated_neg_embeddings);
    assert!(
        !negative_embeddings.is_empty(),
        "Wake-word bench negative pool must be non-empty — an all-skip run would \
         silently calibrate a weaker enrollment"
    );

    let enrollment: Option<crate::audio::wake_word::WakeWordEnrollment> =
        match super::enrollment_consistency_check(&utterance_embeddings) {
            Ok(proto) => {
                let calibration =
                    crate::audio::wake_word::calibrate_negatives(&proto, &negative_embeddings);
                let created_at = crate::db::now();
                let trained_at = crate::db::now();
                let phrase = super::normalize_phrase(BENCH_WAKE_PHRASE);
                crate::audio::wake_word::WakeWordEnrollment::build(
                    phrase,
                    &utterance_embeddings,
                    calibration,
                    &negative_embeddings,
                    created_at,
                    trained_at,
                )
            }
            Err(err) => {
                warn!("enrollment_consistency_check FAILED: {err} — no enrollment to evaluate");
                None
            }
        };
    let Some(enrollment) = enrollment else {
        eprintln!(
            "FATAL: no enrollment (consistency gate failed) — cannot run the wake-word bench"
        );
        return;
    };
    info!(
        "Enrollment built: phrase='{}', {} utterances, calibration neg_mean={:.4} (p99={:.4}, n={})",
        enrollment.phrase,
        enrollment.utterance_count,
        enrollment.calibration.neg_mean,
        enrollment.calibration.neg_p99,
        enrollment.calibration.n_negatives,
    );
    super::set_enrollment(enrollment);

    // ── Metric 1: Recognition — the 40-clip held-out wake-only basis ──
    // Same generation and the SAME real streaming cold pass
    // (run_enrolled_cold_variant → run_streaming_detection →
    // handle_wake_word_detection).
    let held_out_recall_clips = generate_held_out_recall_clips_cached(
        &voice_allocation.enrolled,
        &model_version_hash,
        &cache_dir_path,
    );
    info!(
        "Recognition basis: {} held-out wake-only clips (enrolled voice, seeds 3000+)",
        held_out_recall_clips.len(),
    );
    let mut recognized = 0usize;
    for (pcm, _label) in &held_out_recall_clips {
        if run_enrolled_cold_variant(pcm) {
            recognized += 1;
        }
    }
    let recognition_total = held_out_recall_clips.len();
    let recognition_rate = if recognition_total > 0 {
        recognized as f64 / recognition_total as f64
    } else {
        f64::NAN
    };
    info!(
        "Recognition: {recognized}/{recognition_total} ({:.1}%)",
        recognition_rate * 100.0,
    );
    eprintln!("  Recognition: {recognized}/{recognition_total} wake-word utterances recognized");

    // ── Metric 2a: False reactions — the 113 non-phrase set ──
    // Shared negative corpus + the shared warm detection pass
    // (test_detection_samples).
    let negative_corpus =
        build_negative_corpus(&available_styles, &model_version_hash, &cache_dir_path);
    let non_phrase_total = negative_corpus.confusable.len()
        + negative_corpus.unrelated.len()
        + negative_corpus.silence.len()
        + negative_corpus.noise.len();
    if non_phrase_total != 113 {
        warn!(
            "Wake-word bench non-phrase set size is {non_phrase_total}, not the pinned 113 \
             (likely TTS synthesis misses on a cold cache) — reporting the actual count",
        );
    }
    let mut fa_metrics = DetectionMetrics::default();
    let mut shared_adaptive = super::AdaptiveThresholdState::warmed();
    test_detection_samples(
        &negative_corpus.confusable,
        &mut fa_metrics,
        |m, l| m.false_accepts.push(l.to_string()),
        Some(&mut shared_adaptive),
        false, // warm pass only (negative phase)
    );
    test_detection_samples(
        &negative_corpus.unrelated,
        &mut fa_metrics,
        |m, l| m.false_accepts.push(l.to_string()),
        Some(&mut shared_adaptive),
        false, // warm pass only (negative phase)
    );
    test_detection_samples(
        &negative_corpus.silence,
        &mut fa_metrics,
        |m, l| m.false_accepts.push(l.to_string()),
        Some(&mut shared_adaptive),
        false, // warm pass only (negative phase)
    );
    test_detection_samples(
        &negative_corpus.noise,
        &mut fa_metrics,
        |m, l| m.false_accepts.push(l.to_string()),
        Some(&mut shared_adaptive),
        false, // warm pass only (negative phase)
    );
    let false_reactions = fa_metrics.false_accepts.len();
    let non_phrase_rate = if non_phrase_total > 0 {
        false_reactions as f64 / non_phrase_total as f64
    } else {
        f64::NAN
    };
    info!(
        "False reactions: {false_reactions}/{non_phrase_total} ({:.1}%)",
        non_phrase_rate * 100.0
    );
    eprintln!("  False reactions: {false_reactions}/{non_phrase_total} on the non-phrase set");

    // ── Metric 2b: Real-audio FA/h (parallel, pinned subset) ──
    let real_audio = run_real_audio_phase();
    // Coverage numbers are read from the real-audio section BEFORE the json!
    // macro moves `real_audio` into the report.
    let real_audio_audio_hours = real_audio["feed"]["audio_hours_fed"].as_f64();
    let real_audio_speech_hours = real_audio["feed"]["speech_hours_fed"].as_f64();
    let real_audio_noise_hours = real_audio["feed"]["noise_hours_fed"].as_f64();

    // ── Report: the three metrics + coverage + worker count + wall time ──
    let wall_clock_secs = overall_start.elapsed().as_secs_f64();
    let report = serde_json::json!({
        "benchmark": "wake_word",
        "wake_phrase": BENCH_WAKE_PHRASE,
        "recognition": {
            "detected": recognized,
            "of": recognition_total,
            "rate": recognition_rate,
            "basis": "existing 40-clip held-out wake-only basis (enrolled voice, \
                      seeds 3000+, fixed bench phrase)",
        },
        "false_reactions": {
            "non_phrase_set": {
                "false_reactions": false_reactions,
                "of": non_phrase_total,
                "rate": non_phrase_rate,
                "basis": "the 113 non-phrase set (56 confusable + 40 unrelated + 3 \
                          silence + 14 noise profiles)",
            },
            "real_audio": real_audio,
        },
        "coverage": {
            "phrase_utterances": recognition_total,
            "non_phrase_set": non_phrase_total,
            "real_audio_audio_hours": real_audio_audio_hours,
            "real_audio_speech_hours": real_audio_speech_hours,
            "real_audio_noise_hours": real_audio_noise_hours,
        },
        "workers": BENCH_WORKERS,
        "wall_clock_secs": wall_clock_secs,
        "fa_per_hour_basis_note": "FA-per-hour denominators are reported as BOTH raw \
                                    audio hours and VAD-active hours (per-worker sums); \
                                    cooldown merge applies per worker.",
        "tts_caveat": "Recognition and the synthetic false-reaction set are measured on \
                       synthesized (TTS) speech, not real human speech; real audio is \
                       used for the false-reaction rate only.",
    });

    // Stop the heartbeat thread.
    heartbeat_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = heartbeat_handle.join();

    println!("--- BENCHMARK_JSON_BEGIN ---");
    let json_text = serde_json::to_string_pretty(&report).expect("JSON serialization");
    println!("{json_text}");
    println!("--- BENCHMARK_JSON_END ---");

    // Persist the report (same directory as the benchmark lock file) so a run
    // survives regardless of how the process ends.
    if let Ok(report_dir) = crate::config::default_config_dir() {
        let report_path = report_dir.join("wake_word_report.json");
        match std::fs::write(&report_path, &json_text) {
            Ok(()) => info!(
                "Wake-word benchmark report written to {}",
                report_path.display()
            ),
            Err(e) => warn!(
                "Could not write wake-word benchmark report to {}: {e}",
                report_path.display()
            ),
        }
    }

    // ── Human-readable report (stderr) ────────────────────────
    let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    eprintln!(
        "\n\
         ═══════════════════════════════════════════════════════════\n\
                 Wake-Word Benchmark Report\n\
         ═══════════════════════════════════════════════════════════\n\
         Date/Time:      {timestamp}\n\
         Wake phrase:    {wake}\n\
         1. Recognition:        {recognized}/{recognition_total} of 40 phrase utterances\n\
         2. False reactions:    {false_reactions}/{non_phrase_total} on the 113 non-phrase set\n\
         \x20  Real-audio FA/h:   see real_audio section ({workers} parallel workers, {assignment})\n\
         3. Coverage:    {recognition_total} utterances + {non_phrase_total} non-phrases + real audio ({audio_hours:.2} h speech+noise)\n\
         Wall time:      {wall:.1}s ({wall_min:.1} min)\n\
         FA/h basis:     raw audio hours AND VAD-active hours (per-worker sums)\n\
         TTS caveat:     recognition + synthetic false reactions are measured on\n\
         \x20               TTS-synthesized speech, not real human speech (real audio\n\
         \x20               feeds the false-reaction rate only)",
        wake = BENCH_WAKE_PHRASE,
        workers = BENCH_WORKERS,
        assignment = BENCH_ASSIGNMENT,
        audio_hours = real_audio_audio_hours.unwrap_or(f64::NAN),
        wall = wall_clock_secs,
        wall_min = wall_clock_secs / 60.0,
    );
}

// ── Fixture tests ─────────────────────────────────────────
// Compile under `cargo test --features voice-tests` (the bench module is
// feature-gated; `#[cfg(test)]` keeps these out of the harness=false bench
// binary).

#[cfg(test)]
mod tests {
    use super::*;

    // ── single-voice enrollment DSP ───────────────────────────

    /// Guided-prompt DSP primitives: determinism, HF attenuation, and group
    /// differentiation.  The filter checks verify the rolloff directly (a
    /// total-energy comparison of the conditioned clips would be dominated by
    /// the level-reduction gains); the condition_enrollment_clip checks cover
    /// determinism, the noise floor, and the morning resample's length shift.
    #[test]
    fn guided_prompt_dsp_deterministic_and_attenuates_hf() {
        let sample_rate = TARGET_SAMPLE_RATE;
        let rms = |x: &[f32]| crate::util::compute_rms(x);

        // one_pole_lowpass: deterministic, attenuates 5 kHz through a 2.2 kHz
        // cutoff, passthrough at Nyquist.
        let hf: Vec<f32> = (0..sample_rate as usize / 4)
            .map(|i| (2.0 * core::f32::consts::PI * 5000.0 * i as f32 / sample_rate as f32).sin())
            .collect();
        let a = one_pole_lowpass(&hf, 2200.0, sample_rate);
        let b = one_pole_lowpass(&hf, 2200.0, sample_rate);
        assert_eq!(a, b, "lowpass must be deterministic");
        assert!(
            rms(&a) < rms(&hf) * 0.5,
            "5 kHz through a 2.2 kHz lowpass must be attenuated"
        );
        assert_eq!(
            one_pole_lowpass(&hf, sample_rate as f32 * 0.5, sample_rate),
            hf,
            "cutoff at Nyquist must pass through unchanged"
        );

        // high_shelf_cut: deterministic, tilts 4 kHz down through a −4 dB
        // @ 3 kHz shelf, passthrough at zero gain.
        let tilt: Vec<f32> = (0..sample_rate as usize / 4)
            .map(|i| (2.0 * core::f32::consts::PI * 4000.0 * i as f32 / sample_rate as f32).sin())
            .collect();
        let a = high_shelf_cut(&tilt, -4.0, 3000.0, sample_rate);
        let b = high_shelf_cut(&tilt, -4.0, 3000.0, sample_rate);
        assert_eq!(a, b, "high-shelf cut must be deterministic");
        assert!(
            rms(&a) < rms(&tilt) * 0.7,
            "4 kHz above a -4 dB @ 3 kHz shelf must be attenuated"
        );
        assert_eq!(high_shelf_cut(&tilt, 0.0, 3000.0, sample_rate), tilt);

        // condition_enrollment_clip: deterministic per clip, every clip gets
        // the noise floor, and the morning group's 0.92× resample is longer.
        let mixed: Vec<f32> = (0..sample_rate as usize / 2)
            .map(|i| {
                0.5 * (2.0 * core::f32::consts::PI * 220.0 * i as f32 / sample_rate as f32).sin()
                    + 0.3
                        * (2.0 * core::f32::consts::PI * 4000.0 * i as f32 / sample_rate as f32)
                            .sin()
            })
            .collect();
        for clip_idx in 0..10 {
            let a = condition_enrollment_clip(&mixed, clip_idx);
            let b = condition_enrollment_clip(&mixed, clip_idx);
            assert_eq!(a, b, "clip {clip_idx} conditioning must be deterministic");
            assert_ne!(
                a, mixed,
                "clip {clip_idx} must be conditioned (noise floor)"
            );
        }
        let normal_len = condition_enrollment_clip(&mixed, 0).len();
        let morning_len = condition_enrollment_clip(&mixed, 8).len();
        assert!(
            morning_len > normal_len,
            "morning clip (0.92x slower resample) must be longer than normal"
        );
    }

    /// Voice allocation: standard 10-style set → enrolled=F1, negative pool
    /// F1..M1 (styles outside the pool never train in).
    #[test]
    fn allocate_voices_standard_set() {
        let styles: Vec<String> = (1..=5)
            .map(|i| format!("F{i}.json"))
            .chain((1..=5).map(|i| format!("M{i}.json")))
            .collect();
        let a = allocate_voices(&styles);
        assert_eq!(a.enrolled, "F1.json");
        assert_eq!(a.negative_pool, styles[..6]);
    }

    // ── PCM cache key tests ─────────────────────────────────────────────────

    #[test]
    fn test_pcm_cache_key_determinism() {
        let h = |text, style, seed| pcm_cache_key(text, style, seed, 16000, "test_hash");
        let a = h("hey mahbot", "default", 42);
        let b = h("hey mahbot", "default", 42);
        assert_eq!(a, b, "same inputs must produce same cache key");
    }

    #[test]
    fn test_pcm_cache_key_sensitivity_to_text() {
        let a = pcm_cache_key("hey mahbot", "default", 42, 16000, "hash");
        let b = pcm_cache_key("hey jarvis", "default", 42, 16000, "hash");
        assert_ne!(a, b, "different text must produce different cache keys");
    }

    #[test]
    fn test_pcm_cache_key_sensitivity_to_seed() {
        let a = pcm_cache_key("hey mahbot", "default", 41, 16000, "hash");
        let b = pcm_cache_key("hey mahbot", "default", 42, 16000, "hash");
        assert_ne!(a, b, "different seed must produce different cache keys");
    }

    #[test]
    fn test_pcm_cache_key_sensitivity_to_model_hash() {
        let a = pcm_cache_key("hey mahbot", "default", 42, 16000, "hash_a");
        let b = pcm_cache_key("hey mahbot", "default", 42, 16000, "hash_b");
        assert_ne!(
            a, b,
            "different model hash must produce different cache keys"
        );
    }

    #[test]
    fn test_tts_model_version_hash_is_non_empty() {
        let hash = tts_model_version_hash();
        assert_eq!(hash.len(), 64, "SHA-256 hex is 64 chars");
        assert!(
            hash.chars().all(|c| c.is_ascii_hexdigit()),
            "hash must be hex: {hash}"
        );
    }

    // ── PCM cache read/write tests ─────────────────────────────────────────

    /// Write synthetic PCM data (16 KB) to `path`.
    fn seed_test_pcm(path: &Path) -> u64 {
        let samples: Vec<f32> = vec![0.0; 4096]; // 16 KB
        write_pcm_cache(path, &samples);
        std::fs::metadata(path).map_or(0, |m| m.len())
    }

    #[test]
    fn pcm_cache_read_normal_returns_some() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("a".repeat(64));
        seed_test_pcm(&path);
        assert!(path.exists());

        let result = read_pcm_cache(&path);
        assert!(result.is_some(), "normal read should return cached PCM");
    }
}
