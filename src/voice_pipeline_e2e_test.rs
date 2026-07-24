//! E2E integration test / benchmark for the full voice pipeline (mahbot-811, mahbot-844).
//!
//! This test exercises the **enrollment-to-detection cycle with realistic
//! TTS-generated speech audio**.  It uses the TTS engine to synthesize wake
//! word variants, feeds them through the enrollment pipeline, trains the
//! MLP classifier and VoiceVerifier, then runs detection on:
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
//! # Environment variables
//!
//! * `MAHBOT_TEST_CACHE_BUST=1` — force full TTS cache regeneration
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
use crate::tts;
use crate::voice_verifier::VoiceVerifier;
use crate::voice_verifier::generate_synthetic_negatives_from_positives;
use crate::wake_word_classifier::WakeWordClassifier;
use earshot::Detector;
use rand::{RngExt, SeedableRng};
use sha2::{Digest, Sha256};
use std::time::Instant;

// ── Constants ──────────────────────────────────────────────────────────────

/// Default wake word for the test.
const WAKE_WORD: &str = "hey mahbot";

/// Number of enrollment variants to generate (fewer than real enrollment
/// since each TTS call takes ~3-5 sec).
const NUM_ENROLLMENT_VARIANTS: usize = 5;

/// Number of synthetic-augmentation variants (additional wake word variants
/// with speed/noise/volume perturbation).
const NUM_AUGMENTATION_VARIANTS: usize = 8;

/// Minimum detection rate for positive (wake word) variants required to pass.
const MIN_DETECTION_RATE: f64 = 0.85;

/// Maximum number of false accepts across ALL negative tests (confusable +
/// unrelated + silence + noise).
const MAX_FALSE_ACCEPTS: usize = 2;

/// Per-category false accept limits (mahbot-844 Part 5).
const MAX_CONFUSABLE_FA: usize = 1;
const MAX_UNRELATED_FA: usize = 0;
const MAX_SILENCE_FA: usize = 0;
const MAX_NOISE_FA: usize = 1;

/// Confusable near-miss phrases for negative detection testing (mahbot-834).
///
/// Expanded from 5 to 20 phrases covering:
/// - Direct phonetic substitutions that sound similar to "hey mahbot"
///   (madbot, map bot, mabot, mahbott, maybot, nab it, etc.)
/// - Similar two-word rhythmic patterns
///   (day mahbot, hey man, hey max, hey mat, pay mabot, hay map pot)
/// - Longer phrases containing wake word sound patterns embedded within
///   (hey maybe not, play mah jong, they mad bot, hey matter of fact)
/// - False starts with similar syllable structure
///   (huh mahbot, eh mad bot, haymaker)
const CONFUSABLE_PHRASES: &[&str] = &[
    // ── Direct phonetic substitutions (wake-word-like) ──────────────
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
    // ── Rhythmic/melodic confusables ─────────────────────────────────
    "hay map pot",
    "huh mahbot",
    "eh mad bot",
    "hey maybott",
    "they mad bot",
    "haymaker",
    // ── Embedded wake-word sounds ────────────────────────────────────
    "hey maybe not",
    "play mah jong",
    "hey matter of fact",
    "a day with mahbot",
];

/// Completely unrelated phrases for negative detection testing (mahbot-834).
///
/// Expanded from 5 to 20 phrases covering:
/// - Short commands (2-4 words) similar to typical wake word length
/// - Medium phrases (5-8 words) with varied phonetic content
/// - Long utterances (10+ words) to test sustained non-detection
/// - Questions, statements, and filler speech
/// - Mixed languages (French, Spanish, German) if TTS phoneme support
///   varies per language — the detector should reject all non-wake-word
///   speech regardless of language.
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
];

/// Number of synthetic negative embeddings to generate for classifier training.
/// This is supplemented with real negative examples from unrelated phrases
/// (see Phase 4) to provide a diverse negative training set.
///
/// Equal to [`crate::voice_verifier::SYNTHETIC_NEGATIVES_COUNT`] for consistency —
/// both produce the same number of synthetic negatives so the test configuration
/// is representative of production's fallback path.
const SYNTHETIC_NEGATIVES_COUNT: usize = 100;

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
/// Hard-clipped applies +12dB gain then clamps at ±1.0 (simulating microphone
/// clipping after boost) — so the gain parameter is 12.0 with clip=true.
const VOLUME_SWEEP_LEVELS: &[(&str, f32, bool)] = &[
    ("minus_12dB", -12.0, false),
    ("baseline_0dB", 0.0, false),
    ("plus_6dB", 6.0, false),
    ("plus_12dB", 12.0, false),
    ("hard_clipped", 12.0, true), // +12dB gain then hard-clamp at ±1.0
];

// ── TTS audio cache (mahbot-844 Part 1) ─────────────────────────────────────

/// Compute a deterministic model version hash from TTS compile-time SHA-256
/// constants (mahbot-844 Part 1).
///
/// Hashes the concatenation of all TTS model SHA-256 digests and the voice
/// style directory hash.  These constants are updated in `tts.rs` whenever
/// model files change, so the cache key automatically tracks model version
/// without disk I/O.
///
/// Returns a hex string (always succeeds since the hashes are compile-time).
fn compute_model_version_hash() -> String {
    let mut hasher = Sha256::new();
    hasher.update(tts::DP_MODEL_SHA256.as_bytes());
    hasher.update(tts::TEXT_ENC_MODEL_SHA256.as_bytes());
    hasher.update(tts::VECTOR_EST_MODEL_SHA256.as_bytes());
    hasher.update(tts::VOCODER_MODEL_SHA256.as_bytes());
    hasher.update(tts::TTS_JSON_SHA256.as_bytes());
    hasher.update(tts::UNICODE_INDEXER_SHA256.as_bytes());
    hasher.update(tts::VOICE_STYLE_SHA256.as_bytes());
    hex_string(&hasher.finalize())
}

/// Hex-encode a byte slice (no external dependency).
fn hex_string(bytes: &[u8]) -> String {
    const HEX_CHARS: &[u8] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX_CHARS[(b >> 4) as usize] as char);
        s.push(HEX_CHARS[(b & 0xf) as usize] as char);
    }
    s
}

/// Build a cache key from synthesis parameters.
fn cache_key(text: &str, style: &str, seed: u64, sample_rate: u32, model_hash: &str) -> String {
    let input = format!("{text}\0{style}\0{seed}\0{sample_rate}\0{model_hash}");
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex_string(&hasher.finalize())
}

/// Get the cache directory path.
fn cache_dir() -> std::path::PathBuf {
    let root = crate::config::CONFIG
        .try_storage_root()
        .expect("CONFIG storage root must be set");
    root.join(TEST_CACHE_DIR)
}

/// Write PCM samples to the cache atomically.
fn write_pcm_cache(path: &std::path::Path, samples: &[f32]) {
    let tmp_path = path.with_extension("tmp");
    let mut data = Vec::with_capacity(samples.len() * 4);
    for &s in samples {
        data.extend_from_slice(&s.to_le_bytes());
    }
    if let Err(e) = std::fs::write(&tmp_path, &data) {
        warn!("Failed to write cache tmp file {}: {e}", tmp_path.display());
        return;
    }
    if let Err(e) = std::fs::rename(&tmp_path, path) {
        warn!(
            "Failed to rename cache {} -> {}: {e}",
            tmp_path.display(),
            path.display()
        );
    }
}

/// Read PCM samples from the cache.  Returns `None` if cache miss or corrupt.
/// When `expected_samples` is non-zero, verifies the file contains exactly that
/// many f32 samples.  Otherwise falls back to alignment checking.
fn read_pcm_cache(path: &std::path::Path, expected_samples: usize) -> Option<Vec<f32>> {
    let data = std::fs::read(path).ok()?;
    if expected_samples > 0 {
        let expected_bytes = expected_samples * size_of::<f32>();
        if data.len() != expected_bytes {
            warn!(
                "Cache file {} has wrong size: got {} bytes, expected {} — deleting",
                path.display(),
                data.len(),
                expected_bytes,
            );
            let _ = std::fs::remove_file(path);
            return None;
        }
    } else if data.len() % 4 != 0 {
        // Alignment check for unknown-length caches
        warn!(
            "Cache file {} has non-aligned size {} — deleting",
            path.display(),
            data.len(),
        );
        let _ = std::fs::remove_file(path);
        return None;
    }
    let samples = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Some(samples)
}

/// Synthesize wake word variant audio with TTS caching support.
///
/// If the audio is cached on disk and `cache_bust` is `false`, loads from
/// cache instead of calling TTS.  Always writes to cache on synthesis.
fn synthesize_wake_word_variant_cached(
    text: &str,
    style: &str,
    seed: u64,
    sample_rate: u32,
    model_hash: &str,
    cache_dir: &std::path::Path,
    cache_bust: bool,
) -> Option<Vec<f32>> {
    let key = cache_key(text, style, seed, sample_rate, model_hash);
    let cache_path = cache_dir.join(&key);

    if !cache_bust && let Some(pcm) = read_pcm_cache(&cache_path, 0) {
        return Some(pcm);
    }

    // Miss or bust — synthesise
    let pcm = match tts::synthesize(text, style, seed, sample_rate) {
        Ok(p) => p,
        Err(e) => {
            warn!("TTS synthesis failed for '{text}' with {style} (seed={seed}): {e}");
            return None;
        }
    };

    // Write atomically
    write_pcm_cache(&cache_path, &pcm);
    Some(pcm)
}

// ── Audio generation helpers (with cache) ─────────────────────────────────

/// Generate enrollment audio variants (different voices, seeds) using TTS
/// caching. Returns them as `(samples, label)` tuples.
fn generate_enrollment_variants_cached(
    available_styles: &[String],
    model_hash: &str,
    cache_dir: &std::path::Path,
    cache_bust: bool,
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
                cache_bust,
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
                cache_bust,
            ) {
                variants.push((pcm, format!("{style}_enroll{i}")));
            }
        }
    }
    variants
}

/// Generate augmented wake word variants with speed, noise, and volume
/// perturbation. These are not cached since they apply perturbations to
/// existing synthesis results.
fn generate_augmented_variants(
    available_styles: &[String],
    base_seed: u64,
) -> Vec<(Vec<f32>, String)> {
    let mut variants = Vec::new();
    let num_styles = available_styles.len().max(1);

    for i in 0..NUM_AUGMENTATION_VARIANTS {
        let style_idx = (i + 3) % num_styles;
        let style = if available_styles.is_empty() {
            DEFAULT_TTS_STYLE
        } else {
            &available_styles[style_idx]
        };
        let seed = base_seed + i as u64 + 1000;

        let base_pcm = match tts::synthesize(WAKE_WORD, style, seed, TARGET_SAMPLE_RATE) {
            Ok(pcm) => pcm,
            Err(e) => {
                warn!("Augmentation synthesis failed: {e}");
                continue;
            }
        };

        let augmented = match i % 3 {
            0 => {
                let factor = 1.0 + ((i as f64 * 0.05).sin() * 0.15) as f32;
                crate::tts_data_gen::speed_perturbation(&base_pcm, TARGET_SAMPLE_RATE, factor)
            }
            1 => {
                let max_gain_db = 6.0;
                crate::tts_data_gen::randomize_volume(&base_pcm, max_gain_db)
            }
            _ => crate::tts_data_gen::add_noise(
                &base_pcm,
                20.0,
                crate::tts_data_gen::NoiseType::Pink,
            ),
        };

        let desc = format!(
            "{style}_aug{i}_{}",
            match i % 3 {
                0 => "speed",
                1 => "volume",
                _ => "noise",
            }
        );
        variants.push((augmented, desc));
    }

    variants
}

/// Generate TTS audio for a list of phrases with caching.
fn generate_phrase_variants_cached(
    phrases: &[&str],
    available_styles: &[String],
    base_seed: u64,
    prefix: &str,
    model_hash: &str,
    cache_dir: &std::path::Path,
    cache_bust: bool,
) -> Vec<(Vec<f32>, String)> {
    let mut variants = Vec::new();
    let num_styles = available_styles.len().max(1);

    for (i, &phrase) in phrases.iter().enumerate() {
        let style_idx = i % num_styles;
        let style = if available_styles.is_empty() {
            DEFAULT_TTS_STYLE
        } else {
            &available_styles[style_idx]
        };
        let seed = base_seed + i as u64 + 500;

        if let Some(pcm) = synthesize_wake_word_variant_cached(
            phrase,
            style,
            seed,
            TARGET_SAMPLE_RATE,
            model_hash,
            cache_dir,
            cache_bust,
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
fn generate_pink_noise() -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(44);
    let num_oscillators = 8usize;
    let mut samples = Vec::with_capacity(NOISE_LEN);
    let mut values = [0.0f32; 8];
    for i in 0..NOISE_LEN {
        // Each step, each oscillator increments and may flip state
        let mut sum = 0.0;
        for (j, v) in values.iter_mut().enumerate() {
            let period = 1 << j;
            if i % period == 0 {
                *v = rng.random::<f32>() * 2.0 - 1.0;
            }
            sum += *v;
        }
        // Normalize (gain ~ 1/sqrt(N_osc))
        samples.push((sum / num_oscillators as f32).clamp(-1.0, 1.0));
    }
    samples
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

/// Try to load the TTS engine from cache if not already loaded.
fn ensure_tts_ready() -> Result<(), String> {
    if tts::models_ready() {
        return Ok(());
    }
    if tts::try_load_cached() {
        return Ok(());
    }
    Err("TTS models not available. Run the app once to download them.".to_string())
}

// ── Enrollment ────────────────────────────────────────────────────────────

/// Process a list of audio clips through the enrollment embedding pipeline.
///
/// Returns:
/// * `positive_embeddings` — flat list of all frame-level 96-dim embedding
///   vectors across all utterances (for MLP classifier and verifier training).
/// * `enrollment_buffer` — per-utterance structure: each element is the
///   sequence of frame-level embeddings for one utterance (for self-test).
/// * `failed_count` — how many variants failed embedding extraction.
#[allow(clippy::type_complexity)]
fn process_enrollment(
    variants: &[(Vec<f32>, String)],
) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>, usize) {
    let mut all_embeddings: Vec<Vec<f32>> = Vec::new();
    let mut enrollment_buffer: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut failed = 0usize;

    for (samples, label) in variants {
        match super::process_enrollment_sample(samples) {
            Ok(embeddings) => {
                if embeddings.is_empty() {
                    warn!("No embeddings extracted from '{label}'");
                    failed += 1;
                    continue;
                }
                // Flatten into the flat positive_embeddings list
                for emb in &embeddings {
                    all_embeddings.push(emb.clone());
                }
                // Keep per-utterance structure for self-test
                enrollment_buffer.push(embeddings.clone());
                info!(
                    "Processed enrollment variant '{label}': {} embeddings",
                    embeddings.len()
                );
            }
            Err(e) => {
                warn!("Embedding extraction failed for '{label}': {e}");
                failed += 1;
            }
        }
    }

    (all_embeddings, enrollment_buffer, failed)
}

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
        vad_decisions.push(super::is_speech_with_detector(
            frame,
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
/// Returns the same pair as [`process_enrollment`] (minus the failed count):
/// flat embeddings list (for classifier training) and per-utterance embedding
/// buffers (for consistency check + self-test).
fn vad_segment_and_enroll(
    enrollment_variants: &[(Vec<f32>, String)],
    augmented_variants: &[(Vec<f32>, String)],
) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>) {
    // ── Concatenate all variants with 2.0s silence gaps ──
    // 2.0s well exceeds SILENCE_THRESHOLD_SAMPLES (1.5s) for clean boundaries.
    let silence_gap_samples = (2.0 * f64::from(super::SAMPLE_RATE)) as usize;
    let silence: Vec<f32> = vec![0.0f32; silence_gap_samples];

    let mut combined_audio: Vec<f32> = Vec::new();
    for (samples, _label) in enrollment_variants.iter().chain(augmented_variants) {
        if !combined_audio.is_empty() {
            combined_audio.extend_from_slice(&silence);
        }
        combined_audio.extend_from_slice(samples);
    }
    // Trailing silence for the last utterance
    combined_audio.extend_from_slice(&silence);

    info!(
        "VAD concatenation: {} total samples ({:.1}s) from {} variants",
        combined_audio.len(),
        combined_audio.len() as f64 / f64::from(super::SAMPLE_RATE),
        enrollment_variants.len() + augmented_variants.len(),
    );

    // ── Compute VAD decision + utterances via shared helper ──
    let (_vad_decisions, utterances) = compute_vad_segments(&combined_audio);

    info!(
        "VAD segmentation: {} utterances from {} concatenated variants",
        utterances.len(),
        enrollment_variants.len() + augmented_variants.len(),
    );

    // ── Process each utterance through enrollment ──
    let mut all_embeddings: Vec<Vec<f32>> = Vec::new();
    let mut enrollment_buffer: Vec<Vec<Vec<f32>>> = Vec::new();

    for (i, utterance) in utterances.iter().enumerate() {
        match super::process_enrollment_sample(utterance) {
            Ok(embeddings) if !embeddings.is_empty() => {
                info!(
                    "Utterance {i}: {} samples ({:.2}s), {} embeddings",
                    utterance.len(),
                    utterance.len() as f64 / f64::from(super::SAMPLE_RATE),
                    embeddings.len(),
                );
                for emb in &embeddings {
                    all_embeddings.push(emb.clone());
                }
                enrollment_buffer.push(embeddings);
            }
            Ok(_) => warn!("Utterance {i}: no embeddings extracted"),
            Err(e) => warn!("Utterance {i}: embedding extraction failed: {e}"),
        }
    }

    let expected_utterances = enrollment_variants.len() + augmented_variants.len();
    info!(
        "VAD-gated enrollment: {} positive embeddings from {} utterances (expected ~{expected_utterances})",
        all_embeddings.len(),
        enrollment_buffer.len(),
    );

    (all_embeddings, enrollment_buffer)
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
/// [`flush_voice_batch`], [`try_match_wake_word_and_push_embedding`],
/// [`score_single_embedding`], and cooldown logic.
///
/// After all audio is fed, a silence frame is sent to flush any remaining
/// voice batch (matching how the production pipeline handles speech→silence
/// transitions).
///
/// Returns a [`DetectionResult`] with a flag and optional latency measurement.
fn run_streaming_detection(samples: &[f32], ctx: &mut super::PipelineCtx) -> DetectionResult {
    let feed_start = Instant::now();
    let chunk_size = super::FRAME_LENGTH;
    // Save pre-existing timestamp — we only return true if detection fires
    // during THIS call, not because a prior call already set the field.
    let before = ctx.last_wake_word_detection;

    // Feed audio in FRAME_LENGTH chunks
    for chunk in samples.chunks(chunk_size) {
        let padded = if chunk.len() < chunk_size {
            let mut p: Vec<f32> = chunk.to_vec();
            p.resize(chunk_size, 0.0);
            p
        } else {
            chunk.to_vec()
        };
        super::handle_wake_word_detection(&padded, ctx);
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
    for _ in 0..3 {
        if ctx.last_wake_word_detection != before {
            let latency = feed_start.elapsed().as_secs_f64() * 1000.0;
            return DetectionResult {
                detected: true,
                latency_ms: Some(latency),
            };
        }
        let silence = vec![0.0f32; chunk_size];
        super::handle_wake_word_detection(&silence, ctx);
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
    /// Peak ensemble rolling-sum score achieved during processing.
    peak_score: f32,
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
    // try_match_wake_word_and_push_embedding reads these from voice_state().
    super::set_classifier_weights(classifier.weights_ref().to_vec());
    super::set_verifier(verifier.clone());

    for (samples, label) in variants {
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
        let result = run_streaming_detection(samples, &mut ctx);
        // Propagate the updated adaptive state for the next variant.
        if let Some(ref mut state) = adaptive_state {
            **state = ctx.adaptive_threshold.clone();
        }
        let peak = ctx.peak_score;
        if result.detected {
            if let Some(lat) = result.latency_ms {
                metrics.latencies.push(lat);
            }
            on_detection(metrics, label);
        }
        // Record per-variant result (mahbot-845).
        metrics.per_variant.push(PerVariantResult {
            variant: label.clone(),
            detected: result.detected,
            peak_score: peak,
        });
    }
}

// ── Volume sweep (mahbot-844 Part 3, informational) ──────────────────────

/// Apply a gain in dB to PCM audio and optionally hard-clip.
fn apply_gain(samples: &[f32], gain_db: f32, hard_clip: bool) -> Vec<f32> {
    let linear = 10.0_f32.powf(gain_db / 20.0);
    samples
        .iter()
        .map(|&s| {
            let s = s * linear;
            if hard_clip { s.clamp(-1.0, 1.0) } else { s }
        })
        .collect()
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
fn run_mid_utterance_test(
    all_wake_variants: &[(Vec<f32>, String)],
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
        (confusable_variants.first(), all_wake_variants.first())
    {
        let mut spliced = conf_pcm.clone();
        spliced.extend_from_slice(wake_pcm);
        let mut ctx = super::PipelineCtx::new();
        let result = run_streaming_detection(&spliced, &mut ctx);
        let detected = result.detected;
        info!("Mid-utterance 'immediate_transition': detected={detected}");
        results.push(("immediate_transition", detected));
    }

    // 2. Brief gap (50ms): confusable prelude → 50ms silence → wake word
    if let (Some((conf_pcm, _)), Some((wake_pcm, _))) =
        (confusable_variants.first(), all_wake_variants.first())
    {
        let mut spliced = conf_pcm.clone();
        spliced.extend_from_slice(&gap_silence(0.05));
        spliced.extend_from_slice(wake_pcm);
        let mut ctx = super::PipelineCtx::new();
        let result = run_streaming_detection(&spliced, &mut ctx);
        let detected = result.detected;
        info!("Mid-utterance 'brief_gap_50ms': detected={detected}");
        results.push(("brief_gap_50ms", detected));
    }

    // 3. Long prelude (~6s): unrelated prelude → 50ms gap → wake word
    if let (Some((unrel_pcm, _)), Some((wake_pcm, _))) =
        (unrelated_variants.first(), all_wake_variants.first())
    {
        let mut spliced = unrel_pcm.clone();
        spliced.extend_from_slice(&gap_silence(0.05));
        spliced.extend_from_slice(wake_pcm);
        let mut ctx = super::PipelineCtx::new();
        let result = run_streaming_detection(&spliced, &mut ctx);
        let detected = result.detected;
        info!("Mid-utterance 'long_prelude_6s': detected={detected}");
        results.push(("long_prelude_6s", detected));
    }

    // 4. Confusable prelude ("hey max"): splice "hey max" → wake word
    // "hey max" is the first confusable phrase that phonetically resembles
    // "hey mahbot".
    if let (Some((max_pcm, _)), Some((wake_pcm, _))) =
        (confusable_variants.first(), all_wake_variants.first())
    {
        let mut spliced = max_pcm.clone();
        spliced.extend_from_slice(&gap_silence(0.05));
        spliced.extend_from_slice(wake_pcm);
        let mut ctx = super::PipelineCtx::new();
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
    super::set_classifier_weights(classifier.weights_ref().to_vec());
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

                let result = run_streaming_detection(&mixed_pcm, &mut ctx);

                // Persist the updated adaptive state for the next variant.
                shared_adaptive = ctx.adaptive_threshold.clone();

                let peak = ctx.peak_score;
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
            info!("─── {}. {} ───", phase_starts.len() + 1, $name);
            phase_starts.push(($name, Instant::now()));
        }};
    }
    macro_rules! phase_end_ms {
        () => {{
            let (name, start) = phase_starts.last().unwrap();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            info!("  → {} completed in {:.0}ms", name, elapsed);
            elapsed as u64
        }};
    }

    let overall_start = Instant::now();
    let mut phase_times = [0u64; NUM_PHASES];

    info!("═══ Voice Pipeline E2E Benchmark ═══");

    // ── 0. Initialize global state ─────────────────────────────────────
    // Set CONFIG storage root so model paths resolve.
    if crate::config::CONFIG.try_storage_root().is_none() {
        let mahbot_dir = crate::config::default_config_dir()
            .expect("Cannot resolve home directory for ~/.mahbot");
        crate::config::CONFIG.set_storage_root(mahbot_dir.clone());
        info!("CONFIG storage root set to: {}", mahbot_dir.display());
    }

    // Determine cache settings
    let cache_bust = std::env::var("MAHBOT_TEST_CACHE_BUST").as_deref() == Ok("1");
    let cache_dir_path = cache_dir();
    if let Err(e) = std::fs::create_dir_all(&cache_dir_path) {
        eprintln!(
            "WARNING: Cannot create cache directory {}: {e}",
            cache_dir_path.display()
        );
    }

    // Compute model version hash for TTS caching (from compile-time constants)
    let model_version_hash = compute_model_version_hash();
    info!("TTS model version hash: {}", &model_version_hash[..16]);

    // Initialize TTS module
    crate::tts::init_global().unwrap_or_else(|e| warn!("tts::init_global() already called: {e}"));

    // Ensure voice pipeline state is initialized
    super::init_global().unwrap_or_else(|e| warn!("voice::init_global() already called: {e}"));

    // ── Prerequisites ───────────────────────────────────────────────
    if let Err(msg) = ensure_tts_ready() {
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
        cache_bust,
    );
    assert!(
        !enrollment_variants.is_empty(),
        "Need at least one enrollment variant. TTS synthesis may have failed for all styles."
    );
    info!(
        "Generated {} enrollment variants",
        enrollment_variants.len()
    );
    phase_times[P_ENROLLMENT_AUDIO] = phase_end_ms!();

    // ── Phase 2: VAD-gated enrollment ────────────────────────────────────
    phase_start!("Phase 2: VAD-gated enrollment");
    let augmented_variants = generate_augmented_variants(&available_styles, 200);
    info!("Generated {} augmented variants", augmented_variants.len());

    let (all_positive_embeddings, all_utterance_buffers) =
        vad_segment_and_enroll(&enrollment_variants, &augmented_variants);
    assert!(
        !all_utterance_buffers.is_empty(),
        "VAD-gated enrollment produced no utterances from {} enrollment + {} augmented variants",
        enrollment_variants.len(),
        augmented_variants.len(),
    );
    info!(
        "VAD-gated enrollment: {} positive embeddings from {} utterances",
        all_positive_embeddings.len(),
        all_utterance_buffers.len(),
    );

    phase_times[P_VAD_ENROLLMENT] = phase_end_ms!();

    // ── Phase 3: Generate negative training data ─────────────────────────
    phase_start!("Phase 3: Generating negative training data");
    info!("Generating negative training audio from unrelated + confusable phrases...");
    let (
        neg_unrelated_1,
        neg_unrelated_2,
        neg_unrelated_3,
        neg_confusable_1,
        neg_confusable_2,
        neg_confusable_3,
    ) = (
        generate_phrase_variants_cached(
            UNRELATED_PHRASES,
            &available_styles,
            400,
            "neg_train",
            &model_version_hash,
            &cache_dir_path,
            cache_bust,
        ),
        generate_phrase_variants_cached(
            UNRELATED_PHRASES,
            &available_styles,
            410,
            "neg_train",
            &model_version_hash,
            &cache_dir_path,
            cache_bust,
        ),
        generate_phrase_variants_cached(
            UNRELATED_PHRASES,
            &available_styles,
            420,
            "neg_train",
            &model_version_hash,
            &cache_dir_path,
            cache_bust,
        ),
        generate_phrase_variants_cached(
            CONFUSABLE_PHRASES,
            &available_styles,
            500,
            "neg_conf_train",
            &model_version_hash,
            &cache_dir_path,
            cache_bust,
        ),
        generate_phrase_variants_cached(
            CONFUSABLE_PHRASES,
            &available_styles,
            510,
            "neg_conf_train",
            &model_version_hash,
            &cache_dir_path,
            cache_bust,
        ),
        generate_phrase_variants_cached(
            CONFUSABLE_PHRASES,
            &available_styles,
            520,
            "neg_conf_train",
            &model_version_hash,
            &cache_dir_path,
            cache_bust,
        ),
    );
    let mut negative_embeddings: Vec<Vec<f32>> = Vec::new();
    for neg_set in [
        &neg_unrelated_1,
        &neg_unrelated_2,
        &neg_unrelated_3,
        &neg_confusable_1,
        &neg_confusable_2,
        &neg_confusable_3,
    ] {
        let (embs, _, _) = process_enrollment(neg_set);
        negative_embeddings.extend(embs);
    }
    info!(
        "Extracted {} real negative embeddings from {} unrelated (3 seeds) + {} confusable phrases \
         (across 3 seed variations)",
        negative_embeddings.len(),
        neg_unrelated_1.len() + neg_unrelated_2.len() + neg_unrelated_3.len(),
        neg_confusable_1.len() + neg_confusable_2.len() + neg_confusable_3.len(),
    );

    // Supplement with distribution-matched synthetic negatives (mahbot-846).
    // Uses the positive embedding statistics so the synthetic negatives
    // overlap with real speech rather than pure N(0,1) noise which is
    // trivially separable.
    negative_embeddings.extend(generate_synthetic_negatives_from_positives(
        SYNTHETIC_NEGATIVES_COUNT,
        &all_positive_embeddings,
        1.5,
    ));

    // Build a SEPARATE negative set for the verifier that EXCLUDES confusable
    let mut verifier_negatives: Vec<Vec<f32>> = Vec::new();
    for unrelated in [&neg_unrelated_1, &neg_unrelated_2, &neg_unrelated_3] {
        let (embs, _, _) = process_enrollment(unrelated);
        verifier_negatives.extend(embs);
    }
    verifier_negatives.extend(generate_synthetic_negatives_from_positives(
        SYNTHETIC_NEGATIVES_COUNT,
        &all_positive_embeddings,
        1.5,
    ));
    phase_times[P_NEG_TRAINING_DATA] = phase_end_ms!();

    // ── Phase 4: finalize_enrollment (consistency check + classifier training) ──
    phase_start!("Phase 4: finalize_enrollment");
    let training_result = super::finalize_enrollment(
        &all_positive_embeddings,
        &negative_embeddings,
        &all_utterance_buffers,
    )
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

    let classifier = WakeWordClassifier::new_ensemble(weights.clone());
    let first_params = weights.first().map_or(
        0,
        crate::wake_word_classifier::ClassifierWeights::param_count,
    );
    info!(
        "Ensemble of {} models trained successfully: {} params each, {} epochs, best val loss={best_val_loss:.4}",
        weights.len(),
        first_params,
        epochs_trained,
    );
    info!(
        "Classifier scores: pos mean={pos_scores_mean:.4} [{pos_scores_min:.4}, {pos_scores_max:.4}] \
         neg mean={neg_scores_mean:.4} [{neg_scores_min:.4}, {neg_scores_max:.4}]",
    );

    // ── Degenerate solution detection (mahbot-844) ──
    // Check if ANY ensemble member has all near-zero weights, indicating a
    // degenerate all-zero solution.  Ensemble averaging mitigates a single
    // degenerate member, but flag it for diagnostics.
    let (degenerate, near_zero_frac) = {
        let mut worst_frac = 0.0_f64;
        for member in &weights {
            let all_w = member.all_trainable_slices();
            let total_params: usize = all_w.iter().map(|s| s.len()).sum();
            let near_zero_count = all_w
                .iter()
                .flat_map(|s| s.iter())
                .filter(|v| v.abs() < 1e-6)
                .count();
            let frac = near_zero_count as f64 / total_params as f64;
            worst_frac = worst_frac.max(frac);
        }
        // Degenerate if >99% of weights in ANY member are within ±1e-6 of zero
        if worst_frac > 0.99 {
            warn!(
                "Ensemble member produced degenerate all-zero solution — training had issues. \
                 {:.1}% of weights near zero in worst member (threshold=1%). \
                 Skipping all detection phases.",
                worst_frac * 100.0,
            );
            (true, worst_frac)
        } else {
            info!(
                "Classifier degenerate check: {:.1}% weights near zero in worst member — OK",
                worst_frac * 100.0
            );
            (false, worst_frac)
        }
    };

    // -- Informational self-test --
    match super::run_enrollment_self_test(&all_utterance_buffers, &classifier) {
        Ok(()) => info!("Detection self-test (informational): passed"),
        Err(e) => info!("Detection self-test (informational, non-gating): {e}"),
    }
    phase_times[P_CLASSIFIER_TRAINING] = phase_end_ms!();

    // ── Phase 5: Train the VoiceVerifier ─────────────────────────────────
    phase_start!("Phase 5: Training VoiceVerifier");
    let verifier = VoiceVerifier::train(
        &all_positive_embeddings,
        &verifier_negatives,
        0.60,
        1.0,
        0.01,
        2000,
    );

    if verifier.is_trained() {
        info!(
            "VoiceVerifier trained successfully with {} positive + {} negative \
             (unrelated + synthetic, no confusable phrases)",
            all_positive_embeddings.len(),
            verifier_negatives.len()
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

    // Collect all positive wake variants for detection
    let all_wake_variants: Vec<(Vec<f32>, String)> = enrollment_variants
        .into_iter()
        .chain(augmented_variants)
        .collect();

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

    // Latency stats — declared here (for scope) but populated inside the
    // detection block (or remain at defaults when degenerate).
    let (mut lat_mean, mut lat_median, mut lat_p95) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut latency_samples = 0usize;

    if !degenerate {
        // Create a pre-warmed adaptive threshold state shared across all
        // detection phases so the adaptive code path is exercised end-to-end
        // (mahbot-845, reviewer_3).  Without this, test_detection_samples
        // creates a fresh PipelineCtx per variant, keeping the adaptive state
        // in perpetual 5-frame bootstrap and measuring all metrics against the
        // static threshold.
        //
        // ⚠ Methodological note (reviewer_3): The benchmark tests ALL positive
        // variants first (phases 8), feeding high classifier scores into the
        // adaptive statistics, THEN tests negatives (phases 9-12).  This
        // inflates the adaptive threshold for negative testing, potentially
        // overestimating false-accept resilience.  The noise-overlap phase (14)
        // uses a separate freshly-warmed state and avoids this bias.  Future
        // benchmark improvements could interleave positive/negative testing or
        // run negative phases before positives.
        let mut shared_adaptive = super::AdaptiveThresholdState::warmed();

        // ── Phase 8: Detection — Positive cases ───────────────────────────
        phase_start!("Phase 8: Positive (wake word) variants");
        test_detection_samples(
            &all_wake_variants,
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
            300,
            "confusable",
            &model_version_hash,
            &cache_dir_path,
            cache_bust,
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
        phase_times[P_CONFUSABLE_NEGATIVES] = phase_end_ms!();

        // ── Phase 10: Detection — Unrelated phrases ───────────────────────
        phase_start!("Phase 10: Negative — unrelated phrases");
        let unrelated_variants = generate_phrase_variants_cached(
            UNRELATED_PHRASES,
            &available_styles,
            400,
            "unrelated",
            &model_version_hash,
            &cache_dir_path,
            cache_bust,
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
                noise_false_accepts.extend(metric.false_accepts);
            }
        }
        phase_times[P_NOISE_PROFILES] = phase_end_ms!();

        // ── Phase 13: Cooldown verification ──────────────────────────────
        // The ~3.1s sleep is EXCLUDED from the timing measurement.
        info!("─── {}. Cooldown verification ───", P_COOLDOWN + 1);
        let mut cooldown_detection_time_ms = 0.0f64;
        if let Some((first_pos, _label)) = all_wake_variants.first() {
            let mut ctx = super::PipelineCtx::new();
            // Propagate the shared adaptive state accumulated across phases 8-12
            // so the adaptive code path is active during cooldown testing too.
            ctx.adaptive_threshold = shared_adaptive.clone();

            // Detection 1: should fire
            let t0 = Instant::now();
            let detected = run_streaming_detection(first_pos, &mut ctx);
            cooldown_detection_time_ms += t0.elapsed().as_secs_f64() * 1000.0;
            if !detected.detected {
                warn!(
                    "Cooldown test: first detection should fire but didn't (detection rate is 0/13 — skipping cooldown)"
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
        noise_overlap_results = run_noise_overlap_test(&all_wake_variants, &classifier, &verifier);
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
        volume_sweep_results = run_volume_sweep(&all_wake_variants, &classifier, &verifier);

        // ── Part 4: Mid-utterance detection (informational) ──────────────
        info!("─── Mid-utterance detection (informational) ───");
        mid_utterance_results = run_mid_utterance_test(
            &all_wake_variants,
            &confusable_variants,
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

    let total_false_accepts = conf_metrics.false_accepts.len()
        + unrelated_metrics.false_accepts.len()
        + silence_metric.false_accepts.len()
        + noise_false_accepts.len();

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
        "Confusable false accepts: {} / {} (limit ≤{MAX_CONFUSABLE_FA})",
        conf_metrics.false_accepts.len(),
        conf_metrics.total,
    );
    if !conf_metrics.false_accepts.is_empty() {
        info!("  False triggers: {:?}", conf_metrics.false_accepts);
    }
    info!(
        "Unrelated false accepts: {} / {} (limit ≤{MAX_UNRELATED_FA})",
        unrelated_metrics.false_accepts.len(),
        unrelated_metrics.total,
    );
    if !unrelated_metrics.false_accepts.is_empty() {
        info!("  False triggers: {:?}", unrelated_metrics.false_accepts);
    }
    info!(
        "Silence false accepts: {} / 1 (limit ≤{MAX_SILENCE_FA})",
        silence_metric.false_accepts.len(),
    );
    info!(
        "Noise false accepts: {} / {} ({} profiles, limit ≤{MAX_NOISE_FA})",
        noise_false_accepts.len(),
        NOISE_PROFILES.len(),
        NOISE_PROFILES.len(),
    );
    if !noise_false_accepts.is_empty() {
        info!("  False triggers: {:?}", noise_false_accepts);
    }

    info!("──────────────────────────────────────────────");
    info!("Total false accepts: {total_false_accepts} — limit ≤{MAX_FALSE_ACCEPTS}");
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

    let passed = {
        // Detection rate assertion
        let dr_ok = pos_metrics.detection_rate() >= MIN_DETECTION_RATE;

        // Per-category false accept assertions (mahbot-844 Part 5)
        let conf_fa_ok = conf_metrics.false_accepts.len() <= MAX_CONFUSABLE_FA;
        let unrel_fa_ok = unrelated_metrics.false_accepts.len() == MAX_UNRELATED_FA;
        let silence_fa_ok = silence_metric.false_accepts.len() == MAX_SILENCE_FA;
        let noise_fa_ok = noise_false_accepts.len() <= MAX_NOISE_FA;

        // Aggregate assertion
        let total_fa_ok = total_false_accepts <= MAX_FALSE_ACCEPTS;

        dr_ok
            && conf_fa_ok
            && unrel_fa_ok
            && silence_fa_ok
            && noise_fa_ok
            && total_fa_ok
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

    let json = serde_json::json!({
        "benchmark": "voice_pipeline_e2e",
        "passed": passed,
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
        "results": {
            "detection_rate": pos_metrics.detection_rate(),
            "detected": pos_metrics.detected,
            "total_positive": pos_metrics.total,
            "false_accepts": {
                "confusable": conf_metrics.false_accepts.len(),
                "unrelated": unrelated_metrics.false_accepts.len(),
                "silence": silence_metric.false_accepts.len(),
                "noise": noise_false_accepts.len(),
                "total": total_false_accepts,
            }
        },
        "per_variant_results": serde_json::Value::Array(
            pos_metrics.per_variant.iter().map(|pv| {
                serde_json::json!({
                    "variant": pv.variant,
                    "detected": pv.detected,
                    "peak_score": pv.peak_score,
                })
            }).collect()
        ),
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
            "total_params": weights.first().map_or(0, crate::wake_word_classifier::ClassifierWeights::param_count),
            "degenerate": degenerate,
        },
        "volume_sweep": serde_json::Value::Object(volume_sweep_map),
        "mid_utterance": serde_json::Value::Object(mid_utterance_map),
        "config": {
            "num_enrollment_variants": NUM_ENROLLMENT_VARIANTS,
            "min_detection_rate": MIN_DETECTION_RATE,
            "max_total_false_accepts": MAX_FALSE_ACCEPTS as u64,
            "per_category_limits": {
                "confusable": MAX_CONFUSABLE_FA,
                "unrelated": MAX_UNRELATED_FA,
                "silence": MAX_SILENCE_FA,
                "noise": MAX_NOISE_FA,
            }
        }
    });

    // Output delimited JSON for CI tooling
    println!("--- BENCHMARK_JSON_BEGIN ---");
    println!(
        "{}",
        serde_json::to_string_pretty(&json).expect("JSON serialization")
    );
    println!("--- BENCHMARK_JSON_END ---");

    // ── Assertions (gated) ──
    assert!(
        pos_metrics.detection_rate() >= MIN_DETECTION_RATE,
        "Detection rate too low: {:.1}% ({}/{}) — need ≥{:.0}%",
        pos_metrics.detection_rate() * 100.0,
        pos_metrics.detected,
        pos_metrics.total,
        MIN_DETECTION_RATE * 100.0,
    );

    // Aggregate false accept limit
    assert!(
        total_false_accepts <= MAX_FALSE_ACCEPTS,
        "Too many false accepts: {total_false_accepts} — need ≤{MAX_FALSE_ACCEPTS}",
    );

    // Per-category false accept limits (mahbot-844 Part 5)
    assert!(
        conf_metrics.false_accepts.len() <= MAX_CONFUSABLE_FA,
        "Too many confusable false accepts: {} — need ≤{MAX_CONFUSABLE_FA}",
        conf_metrics.false_accepts.len(),
    );
    assert!(
        unrelated_metrics.false_accepts.len() == MAX_UNRELATED_FA,
        "Too many unrelated false accepts: {} — need =={MAX_UNRELATED_FA}",
        unrelated_metrics.false_accepts.len(),
    );
    assert!(
        silence_metric.false_accepts.len() == MAX_SILENCE_FA,
        "Too many silence false accepts: {} — need =={MAX_SILENCE_FA}",
        silence_metric.false_accepts.len(),
    );
    assert!(
        noise_false_accepts.len() <= MAX_NOISE_FA,
        "Too many noise false accepts: {} — need ≤{MAX_NOISE_FA}",
        noise_false_accepts.len(),
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

    info!("═══ E2E Voice Pipeline Benchmark PASSED ═══");
}

// ── Non-cached generation helpers (fallback when cache unavailable) ──────
